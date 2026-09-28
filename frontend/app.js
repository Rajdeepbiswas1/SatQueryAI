const API_BASE = "http://127.0.0.1:8000";


/* =========================================================
   ELEMENT REFERENCES
========================================================= */

const html = document.documentElement;

const themeToggle = document.getElementById("theme-toggle");
const themeIconSun = document.getElementById("theme-icon-sun");
const themeIconMoon = document.getElementById("theme-icon-moon");

const apiStatus = document.getElementById("api-status");

const queryInput = document.getElementById("query");

const modeSingle = document.getElementById("mode-single");
const modeCompare = document.getElementById("mode-compare");

const compareType = document.getElementById("compare-type");
const pairTemporal = document.getElementById("pair-temporal");
const pairOpticalSar = document.getElementById("pair-optical-sar");

const singleUploadArea = document.getElementById("single-upload-area");
const compareUploadArea = document.getElementById("compare-upload-area");

const fileInput1 = document.getElementById("file-input-1");
const fileInputCompare1 = document.getElementById("file-input-compare-1");
const fileInputCompare2 = document.getElementById("file-input-compare-2");

const uploadZone1 = document.getElementById("upload-zone-1");
const uploadZoneCompare1 = document.getElementById("upload-zone-compare-1");
const uploadZoneCompare2 = document.getElementById("upload-zone-compare-2");

const imagePreview1 = document.getElementById("image-preview-1");
const imagePreviewCompare1 = document.getElementById("image-preview-compare-1");
const imagePreviewCompare2 = document.getElementById("image-preview-compare-2");

const previewImg1 = document.getElementById("preview-img-1");
const previewImgCompare1 = document.getElementById("preview-img-compare-1");
const previewImgCompare2 = document.getElementById("preview-img-compare-2");

const previewName1 = document.getElementById("preview-name-1");
const previewNameCompare1 = document.getElementById("preview-name-compare-1");
const previewNameCompare2 = document.getElementById("preview-name-compare-2");

const removeImage1 = document.getElementById("remove-image-1");
const removeImageCompare1 = document.getElementById("remove-image-compare-1");
const removeImageCompare2 = document.getElementById("remove-image-compare-2");

const replaceImage1 = document.getElementById("replace-image-1");
const replaceImageCompare1 = document.getElementById("replace-image-compare-1");
const replaceImageCompare2 = document.getElementById("replace-image-compare-2");

const fileSummary = document.getElementById("file-summary");
const imageryDescription = document.getElementById("imagery-description");
const imageryReady = document.getElementById("imagery-ready");
const actionMessage = document.getElementById("action-message");

const slot1Label = document.getElementById("slot-1-label");
const slot2Label = document.getElementById("slot-2-label");

const analyzeButton = document.getElementById("analyze-button");
const loading = document.getElementById("loading");

const errorPanel = document.getElementById("error-panel");
const errorMessage = document.getElementById("error-message");

const results = document.getElementById("results");
const emptyResult = document.getElementById("empty-result");

const answerElement = document.getElementById("answer");
const confidenceElement = document.getElementById("answer-confidence");

const analysisTask = document.getElementById("analysis-task");
const selectedTool = document.getElementById("selected-tool");
const executionStatus = document.getElementById("execution-status");
const executionTime = document.getElementById("execution-time");

const groundingDetail =
    document.getElementById("grounding-detail");

const groundingModeElement =
    document.getElementById("grounding-mode");

const evidenceSection = document.getElementById("evidence-section");
const evidenceImage = document.getElementById("evidence-image");

const bboxNormalized = document.getElementById("bbox-normalized");
const bboxPixels = document.getElementById("bbox-pixels");
const rgbBands = document.getElementById("rgb-bands");

const executionTrace = document.getElementById("execution-trace");
const fullResponse = document.getElementById("full-response");

const viewerArea = document.getElementById("viewer-area");
const viewerEmpty = document.getElementById("viewer-empty");
const viewerImage = document.getElementById("viewer-image");

const compareViewer = document.getElementById("compare-viewer");
const comparePane1 = document.getElementById("compare-viewer-one");
const comparePane2 = document.getElementById("compare-viewer-two");

const viewerCompareImg1 = document.getElementById("viewer-compare-img-1");
const viewerCompareImg2 = document.getElementById("viewer-compare-img-2");

const viewerEvidence = document.getElementById("viewer-evidence");
const viewerStatus = document.getElementById("viewer-status");

let currentGroundingBBox = null;

const zoomIn = document.getElementById("zoom-in");
const zoomOut = document.getElementById("zoom-out");
const resetView = document.getElementById("reset-view");
const fullscreenView = document.getElementById("fullscreen-view");


/* =========================================================
   STATE
========================================================= */

let currentMode = "single";
let compareMode = "temporal";

let selectedImage1 = null;
let selectedImage2 = null;

let singleViewer = {
    scale: 1,
    offsetX: 0,
    offsetY: 0
};

let compareViewerState = {
    one: {
        scale: 1,
        offsetX: 0,
        offsetY: 0
    },
    two: {
        scale: 1,
        offsetX: 0,
        offsetY: 0
    }
};

let activeComparePane = 1;

let draggingViewer = false;
let dragTarget = "single";
let dragStartX = 0;
let dragStartY = 0;
let dragOriginX = 0;
let dragOriginY = 0;

const MIN_ZOOM = 1;
const MAX_ZOOM = 4;
const ZOOM_STEP = 0.25;


/* =========================================================
   THEME
========================================================= */

function applyTheme(theme) {
    const isLight = theme === "light";

    html.classList.toggle(
        "light-theme",
        isLight
    );

    if (themeIconSun) {
        themeIconSun.classList.toggle(
            "hidden",
            isLight
        );
    }

    if (themeIconMoon) {
        themeIconMoon.classList.toggle(
            "hidden",
            !isLight
        );
    }

    if (themeToggle) {
        const label = isLight
            ? "Switch to dark mode"
            : "Switch to light mode";

        themeToggle.title = label;
        themeToggle.setAttribute(
            "aria-label",
            label
        );
    }

    localStorage.setItem(
        "satquery-theme",
        theme
    );
}


function initializeTheme() {
    const savedTheme =
        localStorage.getItem(
            "satquery-theme"
        );

    if (
        savedTheme === "light" ||
        savedTheme === "dark"
    ) {
        applyTheme(savedTheme);
        return;
    }

    const prefersLight =
        window.matchMedia &&
        window.matchMedia(
            "(prefers-color-scheme: light)"
        ).matches;

    applyTheme(
        prefersLight
            ? "light"
            : "dark"
    );
}


themeToggle?.addEventListener(
    "click",
    () => {
        const nextTheme =
            html.classList.contains(
                "light-theme"
            )
                ? "dark"
                : "light";

        applyTheme(nextTheme);
    }
);


/* =========================================================
   API HEALTH
========================================================= */

async function checkApiStatus() {
    apiStatus.textContent =
        "API: checking...";

    apiStatus.className =
        "status-badge";

    try {
        const response =
            await fetch(
                `${API_BASE}/health`
            );

        if (!response.ok) {
            throw new Error(
                `HTTP ${response.status}`
            );
        }

        const data =
            await response.json();

        if (
            data.status === "ok" ||
            data.status === "online" ||
            data.status === "healthy"
        ) {
            apiStatus.textContent =
                "API Connected";

            apiStatus.classList.add(
                "connected"
            );
        } else {
            apiStatus.textContent =
                "API Available";

            apiStatus.classList.add(
                "connected"
            );
        }

    } catch {
        apiStatus.textContent =
            "API Offline";

        apiStatus.classList.add(
            "failed"
        );
    }
}


/* =========================================================
   QUERY SUGGESTIONS
========================================================= */

document
    .querySelectorAll(".suggestion-chip")
    .forEach((button) => {
        button.addEventListener(
            "click",
            () => {
                const query =
                    button.dataset.query;

                if (!query) {
                    return;
                }

                queryInput.value =
                    query;

                queryInput.focus();
            }
        );
    });


/* =========================================================
   MODE SELECTION
========================================================= */

function setMode(mode) {
    currentMode = mode;

    const isSingle =
        mode === "single";

    modeSingle.classList.toggle(
        "active",
        isSingle
    );

    modeCompare.classList.toggle(
        "active",
        !isSingle
    );

    compareType.classList.toggle(
        "hidden",
        isSingle
    );

    singleUploadArea.classList.toggle(
        "hidden",
        !isSingle
    );

    compareUploadArea.classList.toggle(
        "hidden",
        isSingle
    );

    if (isSingle) {
        imageryDescription.textContent =
            "Add the satellite image you want SatQueryAI to analyse.";
    } else {
        imageryDescription.textContent =
            compareMode === "temporal"
                ? "Add two observations of the same area from different dates."
                : "Add two complementary views of the same area.";
    }

    updatePairLabels();
    updateInterfaceState();
    updateViewerForMode();
    clearResults();
}


function setCompareMode(mode) {
    compareMode = mode;

    pairTemporal.classList.toggle(
        "active",
        mode === "temporal"
    );

    pairOpticalSar.classList.toggle(
        "active",
        mode === "optical-sar"
    );

    if (
        currentMode === "compare"
    ) {
        imageryDescription.textContent =
            mode === "temporal"
                ? "Add two observations of the same area from different dates."
                : "Add two complementary views of the same area.";
    }

    updatePairLabels();
    updateInterfaceState();
    clearResults();
}


function updatePairLabels() {
    if (compareMode === "temporal") {
        slot1Label.textContent =
            "BEFORE";

        slot2Label.textContent =
            "AFTER";

        if (comparePane1) {
            comparePane1
                .querySelector(".viewer-image-label")
                ?.replaceChildren(
                    document.createTextNode("BEFORE")
                );
        }

        if (comparePane2) {
            comparePane2
                .querySelector(".viewer-image-label")
                ?.replaceChildren(
                    document.createTextNode("AFTER")
                );
        }

    } else {

        slot1Label.textContent =
            "OPTICAL";

        slot2Label.textContent =
            "RADAR";

        if (comparePane1) {
            comparePane1
                .querySelector(".viewer-image-label")
                ?.replaceChildren(
                    document.createTextNode("OPTICAL")
                );
        }

        if (comparePane2) {
            comparePane2
                .querySelector(".viewer-image-label")
                ?.replaceChildren(
                    document.createTextNode("RADAR")
                );
        }
    }
}


modeSingle.addEventListener(
    "click",
    () => {
        setMode("single");
    }
);

modeCompare.addEventListener(
    "click",
    () => {
        setMode("compare");
    }
);

pairTemporal.addEventListener(
    "click",
    () => {
        setCompareMode("temporal");
    }
);

pairOpticalSar.addEventListener(
    "click",
    () => {
        setCompareMode("optical-sar");
    }
);


/* =========================================================
   FILE VALIDATION
========================================================= */

function isSupportedImage(file) {
    if (!file) {
        return false;
    }

    const name =
        file.name.toLowerCase();

    return (
        name.endsWith(".tif") ||
        name.endsWith(".tiff") ||
        name.endsWith(".png") ||
        name.endsWith(".jpg") ||
        name.endsWith(".jpeg")
    );
}


function createPreviewUrl(file) {
    return URL.createObjectURL(
        file
    );
}


function revokePreviewUrl(file) {
    if (
        file &&
        file._satqueryPreviewUrl
    ) {
        URL.revokeObjectURL(
            file._satqueryPreviewUrl
        );

        file._satqueryPreviewUrl =
            null;
    }
}


function prepareFile(file) {
    if (!file) {
        return null;
    }

    file._satqueryPreviewUrl =
        createPreviewUrl(file);

    return file;
}

function isTiffFile(file) {
    if (!file) {
        return false;
    }

    const name =
        file.name.toLowerCase();

    return (
        name.endsWith(".tif") ||
        name.endsWith(".tiff")
    );
}


function getFileDisplayUrl(file) {
    if (!file) {
        return "";
    }

    if (file._satqueryDisplayUrl) {
        return file._satqueryDisplayUrl;
    }

    if (!isTiffFile(file)) {
        return file._satqueryPreviewUrl || "";
    }

    return "";
}


async function requestServerPreview(file) {

    if (!isTiffFile(file)) {
        return (
            file?._satqueryPreviewUrl ||
            ""
        );
    }

    const formData =
        new FormData();

    formData.append(
        "image",
        file,
        file.name
    );

    const response =
        await fetch(
            `${API_BASE}/preview`,
            {
                method: "POST",
                credentials: "include",
                body: formData,
            }
        );

    let data = null;

    try {
        data =
            await response.json();

    } catch {
        data = null;
    }

    if (!response.ok) {

        const detail =
            data?.detail ||
            data?.message ||
            `HTTP ${response.status}`;

        throw new Error(
            String(detail)
        );
    }

    const preview =
        data?.visual_preview;

    if (
        !preview ||
        !preview.available ||
        !preview.data_url
    ) {
        throw new Error(
            "The preview service returned no display image."
        );
    }

    return preview.data_url;
}


/* =========================================================
   SINGLE IMAGE
========================================================= */

async function setSingleImage(file) {

    if (!isSupportedImage(file)) {

        showError(
            "Please choose a supported satellite image."
        );

        return;
    }

    revokePreviewUrl(
        selectedImage1
    );

    const targetFile =
        prepareFile(file);

    selectedImage1 =
        targetFile;

    uploadZone1.classList.add(
        "hidden"
    );

    imagePreview1.classList.remove(
        "hidden"
    );

    previewName1.textContent =
        selectedImage1.name;

    updateInterfaceState();

    clearResults();

    if (isTiffFile(targetFile)) {

        previewImg1.removeAttribute(
            "src"
        );

        viewerStatus.textContent =
            "Rendering preview...";

        actionMessage.textContent =
            "Preparing satellite preview...";

        try {

            const displayUrl =
                await requestServerPreview(
                    targetFile
                );

            if (
                selectedImage1 !==
                targetFile
            ) {
                return;
            }

            targetFile._satqueryDisplayUrl =
                displayUrl;

            previewImg1.src =
                displayUrl;

            showSingleImageInViewer(
                displayUrl
            );

            viewerStatus.textContent =
                "Imagery loaded";

            actionMessage.textContent =
                "Image ready. Ask SatQueryAI a question.";

        } catch (error) {

            console.error(
                "TIFF preview failed:",
                error
            );

            showError(
                error instanceof Error
                    ? error.message
                    : "Unable to render TIFF preview."
            );

            clearViewer();
        }

        return;
    }

    previewImg1.src =
        targetFile._satqueryPreviewUrl;

    showSingleImageInViewer(
        targetFile._satqueryPreviewUrl
    );
}


fileInput1.addEventListener(
    "change",
    () => {
        const file =
            fileInput1.files?.[0];

        if (file) {
            setSingleImage(file);
        }

        fileInput1.value = "";
    }
);


removeImage1.addEventListener(
    "click",
    (event) => {
        event.preventDefault();
        event.stopPropagation();

        removeSingleImage();
    }
);


replaceImage1.addEventListener(
    "click",
    (event) => {
        event.preventDefault();
        event.stopPropagation();

        fileInput1.click();
    }
);


function removeSingleImage() {
    revokePreviewUrl(
        selectedImage1
    );

    selectedImage1 =
        null;

    imagePreview1.classList.add(
        "hidden"
    );

    uploadZone1.classList.remove(
        "hidden"
    );

    updateInterfaceState();

    clearViewer();
    clearResults();
}


/* =========================================================
   COMPARE IMAGE 1
========================================================= */

async function setCompareImage1(file) {

    if (!isSupportedImage(file)) {

        showError(
            "Please choose a supported satellite image."
        );

        return;
    }

    revokePreviewUrl(
        selectedImage1
    );

    const targetFile =
        prepareFile(file);

    selectedImage1 =
        targetFile;

    uploadZoneCompare1.classList.add(
        "hidden"
    );

    imagePreviewCompare1.classList.remove(
        "hidden"
    );

    previewNameCompare1.textContent =
        selectedImage1.name;

    updateInterfaceState();
    clearResults();

    if (isTiffFile(targetFile)) {

        previewImgCompare1.removeAttribute(
            "src"
        );

        try {

            const displayUrl =
                await requestServerPreview(
                    targetFile
                );

            if (
                selectedImage1 !==
                targetFile
            ) {
                return;
            }

            targetFile._satqueryDisplayUrl =
                displayUrl;

            previewImgCompare1.src =
                displayUrl;

            updateCompareViewer();

        } catch (error) {

            console.error(
                "TIFF preview failed:",
                error
            );

            showError(
                error instanceof Error
                    ? error.message
                    : "Unable to render TIFF preview."
            );

            updateCompareViewer();
        }

        return;
    }

    previewImgCompare1.src =
        targetFile._satqueryPreviewUrl;

    updateCompareViewer();
}


fileInputCompare1.addEventListener(
    "change",
    () => {
        const file =
            fileInputCompare1.files?.[0];

        if (file) {
            setCompareImage1(file);
        }

        fileInputCompare1.value = "";
    }
);


removeImageCompare1.addEventListener(
    "click",
    (event) => {
        event.preventDefault();
        event.stopPropagation();

        removeCompareImage1();
    }
);


replaceImageCompare1.addEventListener(
    "click",
    (event) => {
        event.preventDefault();
        event.stopPropagation();

        fileInputCompare1.click();
    }
);


function removeCompareImage1() {
    revokePreviewUrl(
        selectedImage1
    );

    selectedImage1 =
        null;

    imagePreviewCompare1.classList.add(
        "hidden"
    );

    uploadZoneCompare1.classList.remove(
        "hidden"
    );

    updateInterfaceState();
    updateCompareViewer();
    clearResults();
}


/* =========================================================
   COMPARE IMAGE 2
========================================================= */

async function setCompareImage2(file) {

    if (!isSupportedImage(file)) {

        showError(
            "Please choose a supported satellite image."
        );

        return;
    }

    revokePreviewUrl(
        selectedImage2
    );

    const targetFile =
        prepareFile(file);

    selectedImage2 =
        targetFile;

    uploadZoneCompare2.classList.add(
        "hidden"
    );

    imagePreviewCompare2.classList.remove(
        "hidden"
    );

    previewNameCompare2.textContent =
        selectedImage2.name;

    updateInterfaceState();
    clearResults();

    if (isTiffFile(targetFile)) {

        previewImgCompare2.removeAttribute(
            "src"
        );

        try {

            const displayUrl =
                await requestServerPreview(
                    targetFile
                );

            if (
                selectedImage2 !==
                targetFile
            ) {
                return;
            }

            targetFile._satqueryDisplayUrl =
                displayUrl;

            previewImgCompare2.src =
                displayUrl;

            updateCompareViewer();

        } catch (error) {

            console.error(
                "TIFF preview failed:",
                error
            );

            showError(
                error instanceof Error
                    ? error.message
                    : "Unable to render TIFF preview."
            );

            updateCompareViewer();
        }

        return;
    }

    previewImgCompare2.src =
        targetFile._satqueryPreviewUrl;

    updateCompareViewer();
}


fileInputCompare2.addEventListener(
    "change",
    () => {
        const file =
            fileInputCompare2.files?.[0];

        if (file) {
            setCompareImage2(file);
        }

        fileInputCompare2.value = "";
    }
);


removeImageCompare2.addEventListener(
    "click",
    (event) => {
        event.preventDefault();
        event.stopPropagation();

        removeCompareImage2();
    }
);


replaceImageCompare2.addEventListener(
    "click",
    (event) => {
        event.preventDefault();
        event.stopPropagation();

        fileInputCompare2.click();
    }
);


function removeCompareImage2() {
    revokePreviewUrl(
        selectedImage2
    );

    selectedImage2 =
        null;

    imagePreviewCompare2.classList.add(
        "hidden"
    );

    uploadZoneCompare2.classList.remove(
        "hidden"
    );

    updateInterfaceState();
    updateCompareViewer();
    clearResults();
}


/* =========================================================
   DRAG AND DROP
========================================================= */

function setupDropZone(
    zone,
    callback
) {
    zone.addEventListener(
        "dragover",
        (event) => {
            event.preventDefault();

            zone.classList.add(
                "dragging"
            );
        }
    );

    zone.addEventListener(
        "dragleave",
        () => {
            zone.classList.remove(
                "dragging"
            );
        }
    );

    zone.addEventListener(
        "drop",
        (event) => {
            event.preventDefault();

            zone.classList.remove(
                "dragging"
            );

            const files =
                Array.from(
                    event.dataTransfer.files || []
                );

            if (
                files.length > 0
            ) {
                callback(
                    files[0]
                );
            }
        }
    );
}


setupDropZone(
    uploadZone1,
    setSingleImage
);

setupDropZone(
    uploadZoneCompare1,
    setCompareImage1
);

setupDropZone(
    uploadZoneCompare2,
    setCompareImage2
);


/* =========================================================
   UI STATE
========================================================= */

function updateInterfaceState() {

    if (
        currentMode === "single"
    ) {

        if (selectedImage1) {

            imageryReady.textContent =
                "READY";

            fileSummary.textContent =
                "1 satellite image selected";

            actionMessage.textContent =
                "Image ready. Ask SatQueryAI a question.";

        } else {

            imageryReady.textContent =
                "WAITING";

            fileSummary.textContent =
                "Add imagery to begin.";

            actionMessage.textContent =
                "Add imagery and ask a question to begin.";
        }

        return;
    }


    const count =
        Number(Boolean(selectedImage1)) +
        Number(Boolean(selectedImage2));


    if (count === 2) {

        imageryReady.textContent =
            "READY";

        fileSummary.textContent =
            "2 images ready";

        actionMessage.textContent =
            compareMode === "temporal"
                ? "Both observations are ready to compare."
                : "Both sensor views are ready.";

    } else if (count === 1) {

        imageryReady.textContent =
            "1 OF 2";

        fileSummary.textContent =
            "Add the second image to continue.";

        actionMessage.textContent =
            "One image is ready. Add the second image.";

    } else {

        imageryReady.textContent =
            "WAITING";

        fileSummary.textContent =
            "Add two related images.";

        actionMessage.textContent =
            "Add the two related images to begin.";
    }
}


/* =========================================================
   ACTIVE COMPARISON PANE
========================================================= */

function setActiveComparePane(pane) {

    activeComparePane =
        pane;

    if (comparePane1) {
        comparePane1.classList.toggle(
            "active-pane",
            pane === 1
        );
    }

    if (comparePane2) {
        comparePane2.classList.toggle(
            "active-pane",
            pane === 2
        );
    }
}


/* =========================================================
   SINGLE VIEW TRANSFORM
========================================================= */

function resetSingleViewerTransform() {

    singleViewer = {
        scale: 1,
        offsetX: 0,
        offsetY: 0
    };

    applyViewerTransforms();
}


/* =========================================================
   COMPARE VIEW TRANSFORM
========================================================= */

function resetCompareViewerTransforms() {

    compareViewerState = {
        one: {
            scale: 1,
            offsetX: 0,
            offsetY: 0
        },

        two: {
            scale: 1,
            offsetX: 0,
            offsetY: 0
        }
    };

    applyViewerTransforms();
}


/* =========================================================
   APPLY TRANSFORMS
========================================================= */

function applyViewerTransforms() {

    if (
        viewerImage &&
        !viewerImage.classList.contains(
            "hidden"
        )
    ) {

        viewerImage.style.transform =
            `translate(
                calc(-50% + ${singleViewer.offsetX}px),
                calc(-50% + ${singleViewer.offsetY}px)
            )
            scale(${singleViewer.scale})`;
    }


    if (viewerCompareImg1) {

        viewerCompareImg1.style.transform =
            `translate(
                ${compareViewerState.one.offsetX}px,
                ${compareViewerState.one.offsetY}px
            )
            scale(${compareViewerState.one.scale})`;
    }


    if (viewerCompareImg2) {

        viewerCompareImg2.style.transform =
            `translate(
                ${compareViewerState.two.offsetX}px,
                ${compareViewerState.two.offsetY}px
            )
            scale(${compareViewerState.two.scale})`;
    }

    if (
        currentGroundingBBox
    ) {
        requestAnimationFrame(
            () => {
                renderGroundingOverlay();
            }
        );
    }
}


/* =========================================================
   GROUNDING OVERLAY
========================================================= */

function normalizeGroundingBBox(value) {

    if (!Array.isArray(value)) {
        return null;
    }

    let bbox = value;

    /*
       Accept either:
           [x1, y1, x2, y2]
       or:
           [[x1, y1, x2, y2]]
    */

    if (
        bbox.length === 1 &&
        Array.isArray(bbox[0])
    ) {
        bbox = bbox[0];
    }

    if (bbox.length !== 4) {
        return null;
    }

    const numbers =
        bbox.map(
            Number
        );

    if (
        numbers.some(
            (x) => !Number.isFinite(x)
        )
    ) {
        return null;
    }

    let [
        x1,
        y1,
        x2,
        y2
    ] = numbers;

    /*
       Keep normalized coordinates inside [0,1].
    */

    x1 = Math.max(
        0,
        Math.min(
            1,
            x1
        )
    );

    y1 = Math.max(
        0,
        Math.min(
            1,
            y1
        )
    );

    x2 = Math.max(
        0,
        Math.min(
            1,
            x2
        )
    );

    y2 = Math.max(
        0,
        Math.min(
            1,
            y2
        )
    );

    if (x2 < x1) {
        [x1, x2] =
            [x2, x1];
    }

    if (y2 < y1) {
        [y1, y2] =
            [y2, y1];
    }

    return {
        x1,
        y1,
        x2,
        y2
    };
}


function removeGroundingOverlays() {

    document
        .querySelectorAll(
            ".satquery-grounding-box"
        )
        .forEach(
            (element) => {
                element.remove();
            }
        );

    viewerEvidence?.classList.add(
        "hidden"
    );
}


function createGroundingBox(
    parent,
    image,
    bbox,
    labelText
) {

    if (
        !parent ||
        !image ||
        !bbox
    ) {
        return;
    }

    const imageRect =
        image.getBoundingClientRect();

    const parentRect =
        parent.getBoundingClientRect();

    if (
        imageRect.width <= 0 ||
        imageRect.height <= 0 ||
        parentRect.width <= 0 ||
        parentRect.height <= 0
    ) {
        return;
    }

    const box =
        document.createElement(
            "div"
        );

    box.className =
        "satquery-grounding-box";

    const left =
        imageRect.left -
        parentRect.left +
        (
            bbox.x1 *
            imageRect.width
        );

    const top =
        imageRect.top -
        parentRect.top +
        (
            bbox.y1 *
            imageRect.height
        );

    const width =
        (
            bbox.x2 -
            bbox.x1
        ) *
        imageRect.width;

    const height =
        (
            bbox.y2 -
            bbox.y1
        ) *
        imageRect.height;

    box.style.left =
        `${left}px`;

    box.style.top =
        `${top}px`;

    box.style.width =
        `${width}px`;

    box.style.height =
        `${height}px`;

    if (labelText) {

        const label =
            document.createElement(
                "span"
            );

        label.className =
            "satquery-grounding-label";

        label.textContent =
            labelText;

        box.appendChild(
            label
        );
    }

    parent.appendChild(
        box
    );
}


function renderGroundingOverlay() {

    removeGroundingOverlays();

    if (
        !currentGroundingBBox ||
        currentMode === null
    ) {
        return;
    }

    const bbox =
        normalizeGroundingBBox(
            currentGroundingBBox
        );

    if (!bbox) {
        return;
    }


    /*
       SINGLE IMAGE
    */

    if (
        currentMode === "single" &&
        !viewerImage.classList.contains(
            "hidden"
        )
    ) {

        createGroundingBox(
            viewerArea,
            viewerImage,
            bbox,
            "GROUNDING"
        );

        viewerStatus.textContent =
            "Grounded region";

        return;
    }


    /*
       CROSS-MODAL
       
       Grounding is drawn on the optical
       observation because the user-facing
       region should be spatially interpretable.
    */

    if (
        currentMode === "compare" &&
        comparePane1 &&
        viewerCompareImg1 &&
        !viewerCompareImg1.classList.contains(
            "hidden"
        )
    ) {

        createGroundingBox(
            comparePane1,
            viewerCompareImg1,
            bbox,
            "GROUNDING"
        );

        viewerStatus.textContent =
            "Grounded region";
    }
}


function setGroundingBBox(
    value
) {

    currentGroundingBBox =
        value || null;

    requestAnimationFrame(
        () => {
            renderGroundingOverlay();
        }
    );
}


window.addEventListener(
    "resize",
    () => {

        if (
            currentGroundingBBox
        ) {
            renderGroundingOverlay();
        }
    }
);

/* =========================================================
   ZOOM
========================================================= */

function zoomSingleViewer(amount) {

    if (!selectedImage1) {
        return;
    }

    singleViewer.scale =
        Math.max(
            MIN_ZOOM,
            Math.min(
                MAX_ZOOM,
                singleViewer.scale + amount
            )
        );


    if (
        singleViewer.scale === MIN_ZOOM
    ) {

        singleViewer.offsetX = 0;
        singleViewer.offsetY = 0;
    }


    applyViewerTransforms();
}


function zoomComparePane(
    pane,
    amount
) {

    const target =
        pane === 1
            ? compareViewerState.one
            : compareViewerState.two;


    target.scale =
        Math.max(
            MIN_ZOOM,
            Math.min(
                MAX_ZOOM,
                target.scale + amount
            )
        );


    if (
        target.scale === MIN_ZOOM
    ) {

        target.offsetX = 0;
        target.offsetY = 0;
    }


    applyViewerTransforms();
}


function zoomViewer(amount) {

    if (
        currentMode === "single"
    ) {

        zoomSingleViewer(
            amount
        );

        return;
    }


    if (
        currentMode === "compare"
    ) {

        zoomComparePane(
            activeComparePane,
            amount
        );
    }
}


/* =========================================================
   VIEWER CONTROLS
========================================================= */

function enableViewerControls() {

    zoomIn.disabled =
        false;

    zoomOut.disabled =
        false;

    resetView.disabled =
        false;
}


function disableViewerControls() {

    zoomIn.disabled =
        true;

    zoomOut.disabled =
        true;

    resetView.disabled =
        true;
}


zoomIn.addEventListener(
    "click",
    (event) => {

        event.preventDefault();

        zoomViewer(
            ZOOM_STEP
        );
    }
);


zoomOut.addEventListener(
    "click",
    (event) => {

        event.preventDefault();

        zoomViewer(
            -ZOOM_STEP
        );
    }
);


resetView.addEventListener(
    "click",
    (event) => {

        event.preventDefault();

        resetSingleViewerTransform();

        resetCompareViewerTransforms();
    }
);


/* =========================================================
   VIEWER MODE
========================================================= */

function showSingleImageInViewer(
    src
) {

    viewerEmpty.classList.add(
        "hidden"
    );

    compareViewer.classList.add(
        "hidden"
    );

    viewerImage.classList.remove(
        "hidden"
    );


    viewerImage.src =
        src;


    resetSingleViewerTransform();

    viewerStatus.textContent =
        "Imagery loaded";

    enableViewerControls();
}


function updateCompareViewer() {

    if (
        currentMode !== "compare"
    ) {
        return;
    }


    viewerImage.classList.add(
        "hidden"
    );


    if (
        selectedImage1 ||
        selectedImage2
    ) {

        viewerEmpty.classList.add(
            "hidden"
        );

        compareViewer.classList.remove(
            "hidden"
        );


        if (selectedImage1) {

            const displayUrl =
                getFileDisplayUrl(
                    selectedImage1
                );

            if (displayUrl) {

                viewerCompareImg1.src =
                    displayUrl;

            } else {

                viewerCompareImg1.removeAttribute(
                    "src"
                );
            }

        } else {

            viewerCompareImg1.removeAttribute(
                "src"
            );
        }


        if (selectedImage2) {

            const displayUrl =
                getFileDisplayUrl(
                    selectedImage2
                );

            if (displayUrl) {

                viewerCompareImg2.src =
                    displayUrl;

            } else {

                viewerCompareImg2.removeAttribute(
                    "src"
                );
            }

        } else {

            viewerCompareImg2.removeAttribute(
                "src"
            );
        }


        resetCompareViewerTransforms();

        setActiveComparePane(
            activeComparePane
        );


        viewerStatus.textContent =
            selectedImage1 &&
            selectedImage2
                ? "Comparison ready"
                : "Awaiting second image";


        enableViewerControls();

    } else {

        clearViewer();
    }
}


function updateViewerForMode() {

    if (
        currentMode === "single"
    ) {

        if (selectedImage1) {

            const displayUrl =
                getFileDisplayUrl(
                    selectedImage1
                );

            if (displayUrl) {

                showSingleImageInViewer(
                    displayUrl
                );

            } else {

                clearViewer();
            }

        } else {

            clearViewer();
        }

    } else {

        updateCompareViewer();
    }
}


function clearViewer() {

    removeGroundingOverlays();

    viewerImage.classList.add(
        "hidden"
    );

    compareViewer.classList.add(
        "hidden"
    );

    viewerEvidence.classList.add(
        "hidden"
    );


    viewerImage.removeAttribute(
        "src"
    );

    viewerCompareImg1.removeAttribute(
        "src"
    );

    viewerCompareImg2.removeAttribute(
        "src"
    );


    viewerEmpty.classList.remove(
        "hidden"
    );


    viewerStatus.textContent =
        "Awaiting input";


    resetSingleViewerTransform();
    resetCompareViewerTransforms();

    disableViewerControls();
}


/* =========================================================
   CLICK TO SELECT COMPARE PANE
========================================================= */

comparePane1?.addEventListener(
    "click",
    () => {
        setActiveComparePane(1);
    }
);


comparePane2?.addEventListener(
    "click",
    () => {
        setActiveComparePane(2);
    }
);


/* =========================================================
   INDEPENDENT WHEEL ZOOM
========================================================= */

comparePane1?.addEventListener(
    "wheel",
    (event) => {

        if (
            currentMode !== "compare" ||
            !selectedImage1
        ) {
            return;
        }

        event.preventDefault();

        setActiveComparePane(1);

        zoomComparePane(
            1,
            event.deltaY < 0
                ? ZOOM_STEP
                : -ZOOM_STEP
        );
    },
    {
        passive: false
    }
);


comparePane2?.addEventListener(
    "wheel",
    (event) => {

        if (
            currentMode !== "compare" ||
            !selectedImage2
        ) {
            return;
        }

        event.preventDefault();

        setActiveComparePane(2);

        zoomComparePane(
            2,
            event.deltaY < 0
                ? ZOOM_STEP
                : -ZOOM_STEP
        );
    },
    {
        passive: false
    }
);


/* =========================================================
   SINGLE VIEW WHEEL ZOOM
========================================================= */

viewerArea.addEventListener(
    "wheel",
    (event) => {

        if (
            currentMode !== "single" ||
            !selectedImage1
        ) {
            return;
        }


        /*
           When comparing, the pane-specific
           handlers above handle the event.
        */

        if (
            currentMode === "compare"
        ) {
            return;
        }


        event.preventDefault();

        zoomSingleViewer(
            event.deltaY < 0
                ? ZOOM_STEP
                : -ZOOM_STEP
        );
    },
    {
        passive: false
    }
);


/* =========================================================
   DRAG / PAN
========================================================= */

function getViewerPointFromEvent(
    event
) {

    if (
        currentMode === "single"
    ) {

        return {
            type: "single",
            state: singleViewer
        };
    }


    const rect =
        comparePane1?.getBoundingClientRect();


    if (
        rect &&
        event.clientX >= rect.left &&
        event.clientX <= rect.right
    ) {

        return {
            type: "compare-one",
            state: compareViewerState.one
        };
    }


    return {
        type: "compare-two",
        state: compareViewerState.two
    };
}


viewerArea.addEventListener(
    "mousedown",
    (event) => {

        if (
            event.button !== 0
        ) {
            return;
        }


        if (
            currentMode === "single"
        ) {

            if (
                !selectedImage1 ||
                singleViewer.scale <= 1
            ) {
                return;
            }

            dragTarget =
                "single";

            dragOriginX =
                singleViewer.offsetX;

            dragOriginY =
                singleViewer.offsetY;

        } else {

            const target =
                getViewerPointFromEvent(
                    event
                );


            if (
                target.type === "compare-one"
            ) {

                if (
                    !selectedImage1 ||
                    compareViewerState.one.scale <= 1
                ) {
                    setActiveComparePane(1);
                    return;
                }

                setActiveComparePane(1);

                dragTarget =
                    "compare-one";

                dragOriginX =
                    compareViewerState.one.offsetX;

                dragOriginY =
                    compareViewerState.one.offsetY;

            } else {

                if (
                    !selectedImage2 ||
                    compareViewerState.two.scale <= 1
                ) {
                    setActiveComparePane(2);
                    return;
                }

                setActiveComparePane(2);

                dragTarget =
                    "compare-two";

                dragOriginX =
                    compareViewerState.two.offsetX;

                dragOriginY =
                    compareViewerState.two.offsetY;
            }
        }


        draggingViewer =
            true;

        dragStartX =
            event.clientX;

        dragStartY =
            event.clientY;

        viewerArea.classList.add(
            "viewer-dragging"
        );

        event.preventDefault();
    }
);


window.addEventListener(
    "mousemove",
    (event) => {

        if (!draggingViewer) {
            return;
        }


        const deltaX =
            event.clientX -
            dragStartX;

        const deltaY =
            event.clientY -
            dragStartY;


        if (
            dragTarget === "single"
        ) {

            singleViewer.offsetX =
                dragOriginX + deltaX;

            singleViewer.offsetY =
                dragOriginY + deltaY;

        } else if (
            dragTarget === "compare-one"
        ) {

            compareViewerState.one.offsetX =
                dragOriginX + deltaX;

            compareViewerState.one.offsetY =
                dragOriginY + deltaY;

        } else {

            compareViewerState.two.offsetX =
                dragOriginX + deltaX;

            compareViewerState.two.offsetY =
                dragOriginY + deltaY;
        }


        applyViewerTransforms();
    }
);


window.addEventListener(
    "mouseup",
    () => {

        draggingViewer =
            false;

        viewerArea.classList.remove(
            "viewer-dragging"
        );
    }
);


/* =========================================================
   FULLSCREEN
========================================================= */

async function toggleFullscreen() {

    try {

        if (
            document.fullscreenElement
        ) {

            await document.exitFullscreen();

            return;
        }


        if (
            viewerArea.requestFullscreen
        ) {

            await viewerArea.requestFullscreen();

            return;
        }


        if (
            viewerArea.webkitRequestFullscreen
        ) {

            viewerArea.webkitRequestFullscreen();

            return;
        }


        showError(
            "Fullscreen viewing is not available in this browser."
        );

    } catch (error) {

        console.error(
            "Fullscreen error:",
            error
        );

        showError(
            "The browser could not open the satellite viewer in fullscreen."
        );
    }
}


fullscreenView.addEventListener(
    "click",
    (event) => {

        event.preventDefault();

        toggleFullscreen();
    }
);


document.addEventListener(
    "fullscreenchange",
    () => {

        if (
            document.fullscreenElement
        ) {

            fullscreenView.textContent =
                "×";

            fullscreenView.title =
                "Exit fullscreen";

        } else {

            fullscreenView.textContent =
                "⛶";

            fullscreenView.title =
                "Fullscreen";
        }
    }
);


/* =========================================================
   RESULTS
========================================================= */

function clearResults() {

    results.classList.add(
        "hidden"
    );

    emptyResult.classList.remove(
        "hidden"
    );

    setGroundingBBox(
        null
    );


    answerElement.textContent =
        "—";

    confidenceElement.textContent =
        "Confidence: N/A";

    analysisTask.textContent =
        "N/A";

    selectedTool.textContent =
        "N/A";

    executionStatus.textContent =
        "N/A";

    executionTime.textContent =
        "N/A";


    executionTrace.textContent =
        "";

    fullResponse.textContent =
        "";


    hideEvidence();
}


function showResults() {

    emptyResult.classList.add(
        "hidden"
    );

    results.classList.remove(
        "hidden"
    );
}


function showError(message) {

    errorMessage.textContent =
        message;

    errorPanel.classList.remove(
        "hidden"
    );
}


function hideError() {

    errorPanel.classList.add(
        "hidden"
    );

    errorMessage.textContent =
        "";
}


function hideEvidence() {

    evidenceSection.classList.add(
        "hidden"
    );

    evidenceImage.removeAttribute(
        "src"
    );

    viewerEvidence.classList.add(
        "hidden"
    );
}


function formatConfidence(value) {

    if (
        value === null ||
        value === undefined ||
        Number.isNaN(
            Number(value)
        )
    ) {
        return "N/A";
    }

    return `${(
        Number(value) * 100
    ).toFixed(1)}%`;
}


function formatValue(value) {

    if (
        value === null ||
        value === undefined
    ) {
        return "N/A";
    }

    if (
        Array.isArray(value)
    ) {
        return JSON.stringify(
            value
        );
    }

    if (
        typeof value === "object"
    ) {
        return JSON.stringify(
            value,
            null,
            2
        );
    }

    return String(value);
}


function formatExecutionTime(value) {

    if (
        value === null ||
        value === undefined
    ) {
        return "N/A";
    }

    const number =
        Number(value);

    if (
        !Number.isFinite(
            number
        )
    ) {
        return String(value);
    }

    return `${number.toFixed(2)} s`;
}


/* =========================================================
   API RESPONSE
========================================================= */

function populateResults(
    data
) {

    const response =
        data.response || {};

    const analysis =
        data.analysis ||
        response.analysis ||
        {};

    const evidenceEnvelope =
        response.evidence ||
        data.evidence ||
        {};

    const evidenceData =
        evidenceEnvelope &&
        typeof evidenceEnvelope === "object" &&
        evidenceEnvelope.data &&
        typeof evidenceEnvelope.data === "object"
            ? evidenceEnvelope.data
            : evidenceEnvelope;

    const execution =
        data.execution_trace ||
        response.execution_trace ||
        {};

    const task =
        analysis.task ??
        data.task ??
        "N/A";

    const isGrounding =
        task === "grounding";


    /* -------------------------------------------------------
       ANSWER
    ------------------------------------------------------- */

    const rawAnswer =
        response.answer ??
        data.answer ??
        evidenceData.answer ??
        null;

    if (isGrounding) {

        answerElement.textContent =
            "Grounded region";

        confidenceElement.textContent =
            "Confidence: N/A";

    } else {

        answerElement.textContent =
            rawAnswer === null
                ? "No answer returned"
                : String(rawAnswer);

        const confidence =
            response.answer_confidence ??
            data.answer_confidence ??
            evidenceData.answer_confidence ??
            evidenceData.answer_probability ??
            null;

        confidenceElement.textContent =
            `Confidence: ${formatConfidence(
                confidence
            )}`;
    }


    /* -------------------------------------------------------
       ANALYSIS DETAILS
    ------------------------------------------------------- */

    analysisTask.textContent =
        task;


    const selectedSpecialist =
        execution.selected_tool ??
        analysis.selected_tool ??
        evidenceData.specialist ??
        evidenceData.selected_specialist ??
        data.selected_tool ??
        "N/A";

    selectedTool.textContent =
        selectedSpecialist;


    executionStatus.textContent =
        data.status ??
        response.status ??
        execution.status ??
        "N/A";


    const executionSeconds =
        data.execution_time_seconds ??
        response.execution_time_seconds ??
        analysis.execution_time_seconds ??
        (
            Number.isFinite(
                Number(execution.total_ms)
            )
                ? Number(execution.total_ms) / 1000
                : null
        );

    executionTime.textContent =
        formatExecutionTime(
            executionSeconds
        );


    /* -------------------------------------------------------
       GROUNDING MODE
    ------------------------------------------------------- */

    if (
        isGrounding &&
        groundingDetail
    ) {

        groundingDetail.classList.remove(
            "hidden"
        );

        if (groundingModeElement) {

            groundingModeElement.textContent =
                String(
                    analysis.grounding_mode ??
                    evidenceData.mode ??
                    evidenceData.grounding_mode ??
                    "reference"
                );
        }

    } else {

        groundingDetail?.classList.add(
            "hidden"
        );

        if (groundingModeElement) {

            groundingModeElement.textContent =
                "N/A";
        }
    }


    /* -------------------------------------------------------
       VISUAL EVIDENCE
    ------------------------------------------------------- */

    const groundingBBox =
        evidenceData.bbox ??
        evidenceData.bbox_normalized ??
        null;
    
    const visualEvidence =
        data.visual_evidence ||
        response.visual_evidence ||
        null;


    if (
        visualEvidence &&
        visualEvidence.available &&
        visualEvidence.data_url
    ) {

        evidenceSection.classList.remove(
            "hidden"
        );

        const displayEvidenceUrl =
            (
                currentMode === "compare" &&
                selectedImage1
            )
                ? getFileDisplayUrl(
                    selectedImage1
                )
                : visualEvidence.data_url;

        evidenceImage.src =
            displayEvidenceUrl;


        bboxNormalized.textContent =
            formatValue(
                visualEvidence.bbox_normalized ??
                evidenceData.bbox
            );


        bboxPixels.textContent =
            formatValue(
                visualEvidence.bbox_pixels
            );


        rgbBands.textContent =
            formatValue(
                visualEvidence.rgb_bands
            );


        if (
            currentMode === "single"
        ) {

            showSingleImageInViewer(
                visualEvidence.data_url
            );

            viewerStatus.textContent =
                "Evidence detected";
        }

    } else if (
        isGrounding &&
        evidenceData &&
        evidenceData.bbox
    ) {

        /*
           Grounding can legitimately return structured
           spatial evidence without a rendered evidence PNG.
           We display the bounding box numerically and do not
           fabricate a visual result.
        */

        bboxNormalized.textContent =
            formatValue(
                evidenceData.bbox
            );

        bboxPixels.textContent =
            "N/A";

        rgbBands.textContent =
            "N/A";

        /*
           Keep the evidence panel visible so the user can
           see that grounding produced spatial evidence.
        */

        evidenceSection.classList.remove(
            "hidden"
        );

        evidenceImage.removeAttribute(
            "src"
        );

    } else {

        hideEvidence();
    }

    if (
        isGrounding &&
        groundingBBox
    ) {

        setGroundingBBox(
            groundingBBox
        );

    } else {

        setGroundingBBox(
            null
        );
    }


    /* -------------------------------------------------------
       TECHNICAL DETAILS
    ------------------------------------------------------- */

    executionTrace.textContent =
        JSON.stringify(
            execution,
            null,
            2
        );

    fullResponse.textContent =
        JSON.stringify(
            data,
            null,
            2
        );


    showResults();
}

/* =========================================================
   ANALYSIS REQUEST
========================================================= */

analyzeButton.addEventListener(
    "click",
    async () => {

        hideError();


        const query =
            queryInput.value.trim();


        if (!query) {

            showError(
                "Please enter a question about your imagery."
            );

            queryInput.focus();

            return;
        }


        const patchIdInput =
            document.getElementById(
                "patch-id"
            );

        const patchId =
            patchIdInput?.value.trim() ||
            "uploaded_scene";


        const optionInputs = [
            document.getElementById("option-a"),
            document.getElementById("option-b"),
            document.getElementById("option-c"),
            document.getElementById("option-d")
        ];


        const optionValues =
            optionInputs.map(
                (input) =>
                    input?.value.trim() || ""
            );


        const anyOptions =
            optionValues.some(
                (value) => value.length > 0
            );


        if (
            anyOptions &&
            !optionValues.every(
                (value) => value.length > 0
            )
        ) {

            showError(
                "Please provide all four options for an MCQ."
            );

            return;
        }


        /* =====================================================
           BUILD REQUEST
        ===================================================== */

        const formData =
            new FormData();

        formData.append(
            "query",
            query
        );

        formData.append(
            "patch_id",
            patchId
        );


        if (anyOptions) {

            formData.append(
                "option_a",
                optionValues[0]
            );

            formData.append(
                "option_b",
                optionValues[1]
            );

            formData.append(
                "option_c",
                optionValues[2]
            );

            formData.append(
                "option_d",
                optionValues[3]
            );
        }


        /* =====================================================
           SINGLE IMAGE
        ===================================================== */

        if (
            currentMode === "single"
        ) {

            if (!selectedImage1) {

                showError(
                    "Please add an image before analysing."
                );

                return;
            }


            const filename =
                selectedImage1.name.toLowerCase();


            if (
                !(
                    filename.endsWith(".tif") ||
                    filename.endsWith(".tiff")
                )
            ) {

                showError(
                    "Single-image analysis currently accepts TIFF/GeoTIFF imagery."
                );

                return;
            }


            formData.append(
                "image_files",
                selectedImage1,
                selectedImage1.name
            );

            formData.append(
                "compare_mode",
                "single"
            );
        }


        /* =====================================================
           TWO IMAGE COMPARISON
        ===================================================== */

        else {

            if (
                !selectedImage1 ||
                !selectedImage2
            ) {

                showError(
                    "Please add both images before comparing them."
                );

                return;
            }


            if (
                compareMode === "optical-sar"
            ) {

                const firstName =
                    selectedImage1.name.toLowerCase();

                const secondName =
                    selectedImage2.name.toLowerCase();


                if (
                    !(
                        (
                            firstName.endsWith(".tif") ||
                            firstName.endsWith(".tiff")
                        ) &&
                        (
                            secondName.endsWith(".tif") ||
                            secondName.endsWith(".tiff")
                        )
                    )
                ) {

                    showError(
                        "Optical + radar analysis currently accepts TIFF/GeoTIFF imagery."
                    );

                    return;
                }


                formData.append(
                    "image_files",
                    selectedImage1,
                    selectedImage1.name
                );

                formData.append(
                    "image_files",
                    selectedImage2,
                    selectedImage2.name
                );

                formData.append(
                    "compare_mode",
                    "optical-sar"
                );

            } else {

                formData.append(
                    "image_files",
                    selectedImage1,
                    selectedImage1.name
                );

                formData.append(
                    "image_files",
                    selectedImage2,
                    selectedImage2.name
                );

                formData.append(
                    "compare_mode",
                    "temporal"
                );
            }
        }


        /* =====================================================
           SEND REQUEST
        ===================================================== */

        analyzeButton.disabled =
            true;

        loading.classList.remove(
            "hidden"
        );

        actionMessage.textContent =
            "SatQueryAI is analysing your imagery...";


        try {

            const response =
                await fetch(
                    `${API_BASE}/analyze`,
                    {
                        method: "POST",
                        credentials: "include",
                        body: formData,
                    }
                );


            let data = null;


            try {

                data =
                    await response.json();

            } catch {

                throw new Error(
                    `Server returned HTTP ${response.status} without a valid JSON response.`
                );
            }


            if (!response.ok) {

                const detail =
                    data?.detail ||
                    data?.message ||
                    `HTTP ${response.status}`;

                throw new Error(
                    String(detail)
                );
            }


            populateResults(
                data
            );


            actionMessage.textContent =
                "Analysis complete.";

        } catch (error) {

            console.error(
                "SatQueryAI analysis error:",
                error
            );


            showError(
                error?.message ||
                "SatQueryAI could not analyse the imagery."
            );


            actionMessage.textContent =
                "Analysis failed. Check the message above.";

        } finally {

            analyzeButton.disabled =
                false;

            loading.classList.add(
                "hidden"
            );
        }
    }
);


/* =========================================================
   INITIALIZATION
========================================================= */

initializeTheme();

setMode(
    "single"
);

clearResults();

disableViewerControls();

checkApiStatus();