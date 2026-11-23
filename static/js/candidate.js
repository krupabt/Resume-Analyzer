/**
 * Resume Analyzer — Candidate Mode Client Script
 * Handles drag-and-drop file upload, file validation, sample JD prefill, and loading states.
 */

document.addEventListener("DOMContentLoaded", () => {
    initSampleJdButton();
    initSingleDropzone();
    initFormSubmission();
});

/**
 * Handle "Load Sample Job Description" button click
 */
function initSampleJdButton() {
    const loadSampleBtn = document.getElementById("loadSampleJdBtn");
    const jdTextarea = document.getElementById("jobDescription");

    if (loadSampleBtn && jdTextarea) {
        loadSampleBtn.addEventListener("click", () => {
            jdTextarea.value = window.SAMPLE_TECH_JD || "";
            jdTextarea.focus();
        });
    }
}

/**
 * Format raw bytes into human-readable string (e.g., 250 KB)
 */
function formatFileSize(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

/**
 * Initialize single resume drag and drop zone
 */
function initSingleDropzone() {
    const dropzone = document.getElementById("singleDropzone");
    const fileInput = document.getElementById("resumeFileInput");
    const previewChip = document.getElementById("filePreviewChip");
    const fileNameDisplay = document.getElementById("fileNameDisplay");
    const fileSizeDisplay = document.getElementById("fileSizeDisplay");
    const removeBtn = document.getElementById("removeFileBtn");
    const dropzoneContent = dropzone ? dropzone.querySelector(".dropzone-content") : null;

    if (!dropzone || !fileInput) return;

    const allowedExtensions = ["pdf", "docx", "txt"];
    const maxSizeBytes = 5 * 1024 * 1024; // 5 MB

    function updatePreview(file) {
        if (!file) return;

        const ext = file.name.split(".").pop().toLowerCase();
        if (!allowedExtensions.includes(ext)) {
            alert(`File "${file.name}" has an unsupported format. Please upload a PDF, DOCX, or TXT file.`);
            clearFile();
            return;
        }

        if (file.size > maxSizeBytes) {
            alert(`File "${file.name}" exceeds the 5MB limit. Please upload a smaller resume file.`);
            clearFile();
            return;
        }

        fileNameDisplay.textContent = file.name;
        fileSizeDisplay.textContent = formatFileSize(file.size);
        previewChip.style.display = "inline-flex";
        if (dropzoneContent) dropzoneContent.style.display = "none";
    }

    function clearFile() {
        fileInput.value = "";
        previewChip.style.display = "none";
        if (dropzoneContent) dropzoneContent.style.display = "block";
    }

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            updatePreview(e.target.files[0]);
        }
    });

    if (removeBtn) {
        removeBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            clearFile();
        });
    }

    // Drag-and-drop events
    ["dragenter", "dragover"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        if (dt && dt.files && dt.files.length > 0) {
            fileInput.files = dt.files;
            updatePreview(dt.files[0]);
        }
    });
}

/**
 * Handle form submit state & spinner
 */
function initFormSubmission() {
    const form = document.getElementById("candidateForm");
    const submitBtn = document.getElementById("submitBtn");

    if (!form || !submitBtn) return;

    form.addEventListener("submit", () => {
        const defaultText = submitBtn.querySelector(".btn-text-default");
        const loadingText = submitBtn.querySelector(".btn-text-loading");

        if (defaultText && loadingText) {
            defaultText.style.display = "none";
            loadingText.style.display = "inline-flex";
        }
        submitBtn.disabled = true;
    });
}
