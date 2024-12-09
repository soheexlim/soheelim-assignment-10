document.getElementById("search-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);

    const response = await fetch("/search", {
        method: "POST",
        body: formData,
    });

    const data = await response.json();
    if (data.error) {
        alert(data.error);
        return;
    }

    const resultsDiv = document.getElementById("results");
    resultsDiv.innerHTML = "";

    data.results.forEach((result) => {
        const div = document.createElement("div");
        div.innerHTML = `
            <img src="/static/${result.file_name}" alt="${result.file_name}" style="width: 150px; height: auto;">
            <p>File: ${result.file_name} | Similarity: ${result.similarity.toFixed(3)}</p>
        `;
        resultsDiv.appendChild(div);
    });
});