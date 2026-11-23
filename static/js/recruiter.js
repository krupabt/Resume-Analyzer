/**
 * Resume Analyzer — Recruiter Mode Client Script
 * Handles multi-file upload (up to 10 resumes), live file list, side-by-side candidate comparison,
 * and loading states.
 */

document.addEventListener("DOMContentLoaded", () => {
    initSampleJdButton();
    initMultiDropzone();
    initFormSubmission();
    initCandidateComparison();
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
 * Format bytes into readable format
 */
function formatFileSize(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

/**
 * Initialize multi-resume upload and live list
 */
function initMultiDropzone() {
    const dropzone = document.getElementById("multiDropzone");
    const fileInput = document.getElementById("resumeFilesInput");
    const fileCountBadge = document.getElementById("fileCountBadge");
    const selectedFilesContainer = document.getElementById("selectedFilesContainer");
    const selectedFilesList = document.getElementById("selectedFilesList");
    const clearAllBtn = document.getElementById("clearAllFilesBtn");

    if (!dropzone || !fileInput) return;

    const MAX_FILES = 10;
    const allowedExtensions = ["pdf", "docx", "txt"];
    let filesArray = [];

    function updateFileListUI() {
        if (!selectedFilesList || !fileCountBadge || !selectedFilesContainer) return;

        selectedFilesList.innerHTML = "";
        const count = filesArray.length;
        fileCountBadge.textContent = `${count} / ${MAX_FILES} Selected`;

        if (count === 0) {
            selectedFilesContainer.style.display = "none";
            return;
        }

        selectedFilesContainer.style.display = "block";

        filesArray.forEach((file, index) => {
            const li = document.createElement("li");
            li.className = "selected-file-row";
            li.innerHTML = `
                <div style="display:flex; align-items:center; gap:0.5rem; overflow:hidden;">
                    <span>📄</span>
                    <strong style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:280px;">${file.name}</strong>
                    <span style="color:var(--text-muted); font-size:0.75rem;">(${formatFileSize(file.size)})</span>
                </div>
                <button type="button" class="btn-remove-file" data-index="${index}" title="Remove file" aria-label="Remove file">&times;</button>
            `;
            selectedFilesList.appendChild(li);
        });

        // Sync DataTransfer to fileInput
        const dataTransfer = new DataTransfer();
        filesArray.forEach(file => dataTransfer.items.add(file));
        fileInput.files = dataTransfer.files;

        // Attach individual remove handlers
        selectedFilesList.querySelectorAll(".btn-remove-file").forEach(btn => {
            btn.addEventListener("click", (e) => {
                const idx = parseInt(e.target.getAttribute("data-index"), 10);
                filesArray.splice(idx, 1);
                updateFileListUI();
            });
        });
    }

    function addFiles(newFiles) {
        for (let i = 0; i < newFiles.length; i++) {
            const file = newFiles[i];
            const ext = file.name.split(".").pop().toLowerCase();

            if (!allowedExtensions.includes(ext)) {
                alert(`File "${file.name}" is not a supported format. Please upload PDF, DOCX, or TXT.`);
                continue;
            }

            if (filesArray.length >= MAX_FILES) {
                alert(`Maximum ${MAX_FILES} resumes allowed. Additional files were not added.`);
                break;
            }

            // Check if duplicate filename already in list
            const isDuplicate = filesArray.some(f => f.name === file.name && f.size === file.size);
            if (!isDuplicate) {
                filesArray.push(file);
            }
        }
        updateFileListUI();
    }

    fileInput.addEventListener("change", (e) => {
        if (e.target.files) {
            addFiles(e.target.files);
        }
    });

    if (clearAllBtn) {
        clearAllBtn.addEventListener("click", () => {
            filesArray = [];
            fileInput.value = "";
            updateFileListUI();
        });
    }

    // Drag-and-drop
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
            addFiles(dt.files);
        }
    });
}

/**
 * Handle form submit state & spinner
 */
function initFormSubmission() {
    const form = document.getElementById("recruiterForm");
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

/**
 * Candidate Side-by-Side Comparison Logic
 */
function initCandidateComparison() {
    const table = document.getElementById("rankingTable");
    const openCompareBtn = document.getElementById("openCompareBtn");
    const compareModal = document.getElementById("compareModal");
    const compareModalBody = document.getElementById("compareModalBody");
    const closeCompareModal = document.getElementById("closeCompareModal");
    const closeCompareFooterBtn = document.getElementById("closeCompareFooterBtn");

    if (!table || !openCompareBtn || !compareModal) return;

    const checkboxes = table.querySelectorAll(".compare-checkbox");

    function getSelectedRows() {
        const selected = [];
        checkboxes.forEach(cb => {
            if (cb.checked) {
                const tr = cb.closest("tr");
                selected.push({
                    name: tr.getAttribute("data-name"),
                    score: tr.getAttribute("data-score"),
                    textScore: tr.getAttribute("data-text-score"),
                    skillScore: tr.getAttribute("data-skill-score"),
                    matched: tr.getAttribute("data-matched") || "None",
                    missing: tr.getAttribute("data-missing") || "None"
                });
            }
        });
        return selected;
    }

    function updateCompareButton() {
        const selected = getSelectedRows();
        openCompareBtn.textContent = `Compare Selected (${selected.length}/2)`;
        openCompareBtn.disabled = selected.length !== 2;
    }

    checkboxes.forEach(cb => {
        cb.addEventListener("change", () => {
            const selected = getSelectedRows();
            if (selected.length > 2) {
                cb.checked = false;
                alert("You can select up to 2 candidates for side-by-side comparison. Deselect one first.");
                return;
            }
            updateCompareButton();
        });
    });

    openCompareBtn.addEventListener("click", () => {
        const selected = getSelectedRows();
        if (selected.length !== 2) return;

        const [c1, c2] = selected;

        compareModalBody.innerHTML = `
            <div class="comparison-grid">
                <div class="compare-candidate-box">
                    <h4 style="font-size:1.2rem; margin-bottom:0.75rem; color:var(--primary);">${c1.name}</h4>
                    <div style="font-size:1.8rem; font-weight:800; margin-bottom:1rem;">${c1.score}% <span style="font-size:0.8rem; color:var(--text-muted); font-weight:500;">Overall Relevance</span></div>
                    <div style="margin-bottom:0.75rem;"><strong>Skill Match Rate:</strong> ${c1.skillScore}%</div>
                    <div style="margin-bottom:0.75rem;"><strong>TF-IDF Text Similarity:</strong> ${c1.textScore}%</div>
                    <div style="margin-bottom:1rem;">
                        <strong style="display:block; margin-bottom:0.35rem; color:var(--success);">Matched Skills:</strong>
                        <p style="font-size:0.88rem;">${c1.matched}</p>
                    </div>
                    <div>
                        <strong style="display:block; margin-bottom:0.35rem; color:var(--warning);">Missing Skills:</strong>
                        <p style="font-size:0.88rem;">${c1.missing}</p>
                    </div>
                </div>

                <div class="compare-candidate-box">
                    <h4 style="font-size:1.2rem; margin-bottom:0.75rem; color:var(--secondary);">${c2.name}</h4>
                    <div style="font-size:1.8rem; font-weight:800; margin-bottom:1rem;">${c2.score}% <span style="font-size:0.8rem; color:var(--text-muted); font-weight:500;">Overall Relevance</span></div>
                    <div style="margin-bottom:0.75rem;"><strong>Skill Match Rate:</strong> ${c2.skillScore}%</div>
                    <div style="margin-bottom:0.75rem;"><strong>TF-IDF Text Similarity:</strong> ${c2.textScore}%</div>
                    <div style="margin-bottom:1rem;">
                        <strong style="display:block; margin-bottom:0.35rem; color:var(--success);">Matched Skills:</strong>
                        <p style="font-size:0.88rem;">${c2.matched}</p>
                    </div>
                    <div>
                        <strong style="display:block; margin-bottom:0.35rem; color:var(--warning);">Missing Skills:</strong>
                        <p style="font-size:0.88rem;">${c2.missing}</p>
                    </div>
                </div>
            </div>
        `;

        compareModal.style.display = "flex";
    });

    const closeModal = () => { compareModal.style.display = "none"; };
    if (closeCompareModal) closeCompareModal.addEventListener("click", closeModal);
    if (closeCompareFooterBtn) closeCompareFooterBtn.addEventListener("click", closeModal);
    compareModal.addEventListener("click", (e) => {
        if (e.target === compareModal) closeModal();
    });
}
