import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ---- meme mapping -----------------------------------------------------
// Each gesture maps to one or more meme images. When a gesture has more
// than one image, one is picked at random each time the gesture is newly
// (re)triggered, so repeated gestures don't always show the same frame.
const GESTURE_MEMES = {
  default: ["memes/sillyhamster.jpeg"],
  heart: ["memes/corazonhamster.jpeg"],
  happy: ["memes/felizhamster.jpeg"],
  gym: ["memes/gymhamster.jpeg"],
  palmsTogether: ["memes/hamster.jpeg"],
  twoFingersTogether: ["memes/muehjejhamster.jpeg"],
  palmsUp: ["memes/nosehamster.jpeg"],
  peace: ["memes/pazhamster.jpeg"],
  pizza: ["memes/pizzahamster.jpeg"],
};

// how many consecutive frames a gesture must hold before we switch to it
const STABLE_FRAMES_REQUIRED = 5;
// MediaPipe landmarks fluctuate a little even when fingertips are touching.
const TOUCH_GAP_THRESHOLD = 1.4;
const PALMS_TOGETHER_MAX_GAP = 2.4;
// if no hand / no gesture is seen for this long, fall back to default
const DEFAULT_FALLBACK_MS = 600;
// how long we trust a stale face box after the face detector loses the face
// (e.g. hand covering the mouth during a shush)
const FACE_STALE_MS = 1200;

const SMILE_SCORE_THRESHOLD = 0.45;
const MOUTH_OPEN_THRESHOLD = 0.03;

// hand-covering-face: how close the hand needs to be to where the mouth
// last was. Wider when the face detector has fully lost the face (strong
// evidence of a real occlusion); tighter when the face is still partially
// tracked (weaker evidence, avoid false positives from a hand just passing
// near the face).
const HAND_COVER_FACE_DIST_FACE_LOST = 1.3;
const HAND_COVER_FACE_DIST_FACE_SEEN = 0.7;

const video = document.getElementById("video");
const memeImg = document.getElementById("memeImg");
const debugHud = document.getElementById("debugHud");

let handLandmarker, faceLandmarker;
let lastVideoTime = -1;
let currentGesture = "default";
let candidateGesture = "default";
let candidateStreak = 0;
let lastNonDefaultAt = performance.now();
let lastFace = null; // { mouthCenter, faceWidth, mouthOpen, smileScore, t }
let lastFaceSeenThisFrame = false;
let lastHandCount = 0;

async function init() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });

  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  });

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  requestAnimationFrame(loop);
}

// ---- 3D-aware geometry helpers -----------------------------------------
// Using z (depth) as well as x/y makes these tests far more robust to hand
// rotation, foreshortening, and motion blur than a plain 2D/wrist-distance
// check would be.
function vec(a, b) {
  return { x: b.x - a.x, y: b.y - a.y, z: (b.z || 0) - (a.z || 0) };
}
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}
function angleDeg(v1, v2) {
  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const m1 = Math.hypot(v1.x, v1.y, v1.z);
  const m2 = Math.hypot(v2.x, v2.y, v2.z);
  if (m1 < 1e-9 || m2 < 1e-9) return 180;
  return (Math.acos(Math.min(1, Math.max(-1, dot / (m1 * m2)))) * 180) / Math.PI;
}

// a finger is "extended" if its two segments (mcp->pip, pip->tip) point in
// roughly the same direction; "curled" if it folds back sharply.
function fingerExtended(lm, mcp, pip, tip) {
  const angle = angleDeg(vec(lm[mcp], lm[pip]), vec(lm[pip], lm[tip]));
  return angle < 45;
}

function classifyHand(lm) {
  const handScale = dist(lm[0], lm[9]) || 1e-6; // wrist -> middle mcp
  const palmSideA = vec(lm[0], lm[5]);
  const palmSideB = vec(lm[0], lm[17]);
  const palmNormal = {
    x: palmSideA.y * palmSideB.z - palmSideA.z * palmSideB.y,
    y: palmSideA.z * palmSideB.x - palmSideA.x * palmSideB.z,
    z: palmSideA.x * palmSideB.y - palmSideA.y * palmSideB.x,
  };
  const palmNormalLength = Math.hypot(palmNormal.x, palmNormal.y, palmNormal.z) || 1e-6;
  // In the "I don't know" shrug, the palms face up toward the ceiling and
  // sit level with or slightly above the wrists.
  const palmFacingUp = Math.abs(palmNormal.y) / palmNormalLength > 0.5 &&
    lm[9].y < lm[0].y + handScale * 0.25;

  const indexUp = fingerExtended(lm, 5, 6, 8);
  const middleUp = fingerExtended(lm, 9, 10, 12);
  const ringUp = fingerExtended(lm, 13, 14, 16);
  const pinkyUp = fingerExtended(lm, 17, 18, 20);

  // thumb + pinky spread apart from each other = shaka/rock-on shape.
  // tucked thumb sits close to the pinky-side of the palm; an abducted
  // thumb sticks straight out and this distance grows a lot.
  const thumbPinkySpread = dist(lm[4], lm[17]) / handScale;
  const thumbOut = thumbPinkySpread > 1.05;

  const curledCount = [indexUp, middleUp, ringUp, pinkyUp].filter((v) => !v).length;

  return {
    indexUp,
    middleUp,
    ringUp,
    pinkyUp,
    thumbOut,
    curledCount,
    handScale,
    indexTip: lm[8],
    wrist: lm[0],
    palmCenter: lm[9],
    thumbTip: lm[4],
    palmFacingUp,
  };
}

function updateFace(faceResult) {
  const now = performance.now();
  const sawFace = !!(faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0);

  if (sawFace) {
    const f = faceResult.faceLandmarks[0];
    const upperLip = f[13];
    const lowerLip = f[14];
    const rightCheek = f[234];
    const leftCheek = f[454];
    const mouthCenter = {
      x: (upperLip.x + lowerLip.x) / 2,
      y: (upperLip.y + lowerLip.y) / 2,
      z: ((upperLip.z || 0) + (lowerLip.z || 0)) / 2,
    };
    const faceWidth = dist(rightCheek, leftCheek);
    // how open the mouth is right now - normalized so it doesn't depend on
    // distance from the camera.
    const mouthOpen = dist(upperLip, lowerLip) / faceWidth;

    const categories = faceResult.faceBlendshapes?.[0]?.categories || [];
    const scoreFor = (name) => categories.find((c) => c.categoryName === name)?.score || 0;
    const smileScore = (scoreFor("mouthSmileLeft") + scoreFor("mouthSmileRight")) / 2;
    lastFace = { mouthCenter, faceWidth, mouthOpen, smileScore, t: now };
  }
  lastFaceSeenThisFrame = sawFace;
}

// a hand is "pointing" if only the index finger is extended (thumb can be
// either way) - the shape both hands make in the finger-tips-touching pose.
function isPointing(h) {
  return h.indexUp && !h.middleUp && !h.ringUp && !h.pinkyUp;
}

function isOpenPalm(h) {
  return h.curledCount === 0;
}

// A relaxed open hand can leave one finger slightly bent in the camera view.
function isMostlyOpenPalm(h) {
  return h.curledCount <= 1;
}

function isPeace(h) {
  return h.indexUp && h.middleUp && !h.ringUp && !h.pinkyUp;
}

// Korean finger heart: index extended, the other fingers folded, and the
// thumb tip touching the index tip.
function isKoreanHeart(h) {
  return !h.middleUp && !h.ringUp && !h.pinkyUp &&
    h.indexTip.y < h.palmCenter.y &&
    dist(h.indexTip, h.thumbTip) / h.handScale < TOUCH_GAP_THRESHOLD;
}

function decideGesture(handResult) {
  const now = performance.now();
  const faceIsFresh = !!lastFace && now - lastFace.t < FACE_STALE_MS;

  if (!handResult.landmarks || handResult.landmarks.length === 0) {
    if (faceIsFresh && lastFace.smileScore > SMILE_SCORE_THRESHOLD && lastFace.mouthOpen > MOUTH_OPEN_THRESHOLD) {
      return "happy";
    }
    return "default";
  }

  const hands = handResult.landmarks.map(classifyHand);

  if (hands.length === 2) {
    const [a, b] = hands;
    const avgScale = (a.handScale + b.handScale) / 2;

    // Muehjej: both index fingertips touch and both thumb tips touch.
    // The other three fingers remain folded, but the index itself may be
    // slightly bent in a natural pose.
    if (a.curledCount >= 3 && b.curledCount >= 3) {
      const tipGap = dist(a.indexTip, b.indexTip) / avgScale;
      const thumbGap = dist(a.thumbTip, b.thumbTip) / avgScale;
      if (tipGap < TOUCH_GAP_THRESHOLD && thumbGap < TOUCH_GAP_THRESHOLD) {
        return "twoFingersTogether";
      }
    }

    // Four extended fingers on each hand, pointing toward the other hand.
    if (isOpenPalm(a) && isOpenPalm(b) &&
        dist(a.indexTip, b.wrist) < dist(a.wrist, b.wrist) &&
        dist(b.indexTip, a.wrist) < dist(a.wrist, b.wrist)) {
      return "pizza";
    }

    // Both palms together (prayer/clap pose).
    if (isMostlyOpenPalm(a) && isMostlyOpenPalm(b) &&
        dist(a.palmCenter, b.palmCenter) / avgScale < PALMS_TOGETHER_MAX_GAP) {
      return "palmsTogether";
    }

    // Both relaxed palms held apart and up, like a shrug. This position
    // test is more reliable than 3D palm orientation from a single camera.
    const palmsApart = dist(a.palmCenter, b.palmCenter) / avgScale >= PALMS_TOGETHER_MAX_GAP;
    if (isMostlyOpenPalm(a) && isMostlyOpenPalm(b) && palmsApart) {
      return "palmsUp";
    }
  }

  const h = hands[0];

  if (isKoreanHeart(h)) {
    return "heart";
  }

  if (faceIsFresh && lastFace.smileScore > SMILE_SCORE_THRESHOLD && lastFace.mouthOpen > MOUTH_OPEN_THRESHOLD) {
    return "happy";
  }

  // Raised fist near the upper face approximates a flexed arm pose.
  if (h.curledCount === 4) {
    if (faceIsFresh && h.palmCenter.y < lastFace.mouthCenter.y - lastFace.faceWidth * 0.25) {
      return "gym";
    }
    return "default";
  }

  if (isPeace(h)) {
    return "peace";
  }

  return "default";
}

function pickImage(gesture) {
  const images = GESTURE_MEMES[gesture];
  return images[Math.floor(Math.random() * images.length)];
}

function applyGesture(gesture) {
  if (gesture === currentGesture) return;
  currentGesture = gesture;
  memeImg.src = pickImage(gesture);
}

function loop() {
  const now = performance.now();
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const ts = performance.now();

    const handResult = handLandmarker.detectForVideo(video, ts);
    lastHandCount = handResult.landmarks?.length || 0;
    const faceResult = faceLandmarker.detectForVideo(video, ts);
    updateFace(faceResult);

    const gesture = decideGesture(handResult);

    // debounce: require a gesture to be seen for several consecutive
    // frames before we commit to it, to avoid flicker between frames
    if (gesture === candidateGesture) {
      candidateStreak++;
    } else {
      candidateGesture = gesture;
      candidateStreak = 1;
    }

    if (candidateStreak >= STABLE_FRAMES_REQUIRED) {
      applyGesture(gesture);
    }

    if (gesture !== "default") lastNonDefaultAt = now;
    if (now - lastNonDefaultAt > DEFAULT_FALLBACK_MS && currentGesture !== "default") {
      applyGesture("default");
    }

    updateDebugHud();
  }
  requestAnimationFrame(loop);
}

function updateDebugHud() {
  if (!debugHud) return;
  debugHud.textContent = `gesture: ${currentGesture}\nhands detected: ${lastHandCount}`;
}

init().catch((err) => console.error(err));
