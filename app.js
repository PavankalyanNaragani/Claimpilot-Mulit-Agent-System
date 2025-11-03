document.addEventListener("DOMContentLoaded", () => {
    // --- DOM Elements ---
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const fileListElem = document.getElementById("file-list");
    const processBtn = document.getElementById("process-btn");
    
    // Status Panel
    const statusTitle = document.querySelector(".status-title");
    const statusSubtitle = document.querySelector(".status-subtitle");
    const statusDetails = document.getElementById("status-details");
    
    // Footer Year
    document.getElementById("year").textContent = new Date().getFullYear();

    // Store selected files in a Map for easy addition/removal
    let selectedFiles = new Map();

    // --- Event Listeners ---
    dropzone.addEventListener("click", () => fileInput.click());
    
    fileInput.addEventListener("change", (e) => {
        handleFiles(e.target.files);
    });

    // Drag and Drop listeners
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        handleFiles(e.dataTransfer.files);
    });

    // Remove file listener (using event delegation)
    fileListElem.addEventListener("click", (e) => {
        if (e.target.classList.contains("remove-file-btn")) {
            const fileName = e.target.dataset.name;
            removeFile(fileName);
        }
    });

    // Process button listener
    processBtn.addEventListener("click", uploadAndProcess);

    // --- Core Functions ---
    function handleFiles(files) {
        for (const file of files) {
            if (file.type === "application/pdf") {
                selectedFiles.set(file.name, file);
            }
        }
        updateFileListUI();
    }

    function removeFile(fileName) {
        selectedFiles.delete(fileName);
        updateFileListUI();
    }

    function updateFileListUI() {
        fileListElem.innerHTML = ""; // Clear list
        if (selectedFiles.size === 0) {
            processBtn.disabled = true;
            return;
        }

        selectedFiles.forEach(file => {
            const li = document.createElement("li");
            li.innerHTML = `
                <span class="file-name" title="${file.name}">${file.name}</span>
                <button class="remove-file-btn" data-name="${file.name}" aria-label="Remove ${file.name}">&times;</button>
            `;
            fileListElem.appendChild(li);
        });
        
        processBtn.disabled = false;
    }

    function setStatus(title, message, isLoading = false, isError = false) {
        statusTitle.textContent = title;
        statusSubtitle.textContent = message;
        statusDetails.innerHTML = "";

        if (isLoading) {
            statusDetails.innerHTML = '<div class="spinner"></div>'; // You can style a spinner later
            processBtn.disabled = true;
            processBtn.textContent = "Processing...";
        } else {
            processBtn.disabled = false;
            processBtn.textContent = "Process Claim";
        }
        
        if (isError) {
            statusTitle.style.color = "var(--danger)";
        } else {
            statusTitle.style.color = "var(--text-color)";
        }
    }

    async function uploadAndProcess() {
        if (selectedFiles.size === 0) {
            alert("Please select files to upload.");
            return;
        }
        
        setStatus("Processing...", "Your documents are being analyzed by the AI agents.", true);

        const formData = new FormData();
        selectedFiles.forEach(file => {
            formData.append("files", file, file.name);
        });

        try {
            // Call the new /process-claim endpoint
            const response = await fetch("/process-claim", {
                method: "POST",
                body: formData,
            });

            const result = await response.json();

            if (!response.ok || result.error) {
                throw new Error(result.error || "An unknown error occurred on the server.");
            }

            // Success! Display the final JSON
            setStatus("Processing Complete", "The claim has been processed successfully.", false);
            statusDetails.textContent = JSON.stringify(result, null, 2);

        } catch (error) {
            console.error("Error processing files:", error);
            setStatus("Error", "Could not process the claim. Please check the console.", false, true);
            statusDetails.textContent = error.message;
        }
    }
});