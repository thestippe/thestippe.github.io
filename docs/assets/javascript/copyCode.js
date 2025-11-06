document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("div.highlight").forEach((block) => {
    const button = document.createElement("button");
    button.className = "copy-code-button";
    button.type = "button";
    button.innerText = "Copy";
    button.title = "Copy code";

    // 🧩 Prevent page scroll/focus jump when clicking inside <pre>
    button.addEventListener("mousedown", (e) => e.preventDefault());

    button.addEventListener("click", async () => {
      const code = block.querySelector("pre code");
      if (!code) return;

      try {
        await navigator.clipboard.writeText(code.innerText);
        button.innerText = "Copied!";
        button.classList.add("copied");

        setTimeout(() => {
          button.innerText = "Copy";
          button.classList.remove("copied");
        }, 2000);
      } catch (err) {
        console.error("Failed to copy code:", err);
      }
    });

    block.style.position = "relative";
    block.appendChild(button);
  });
});
