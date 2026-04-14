chrome.runtime.onMessage.addListener((req, sender, sendResponse) => {
  if (req.action === "captureVisibleTab") {
    chrome.tabs.captureVisibleTab((imageUri) => {
      sendResponse({ imageUri });
    });
    return true; // Keep message channel open for async response
  }

  if (req.action === "translateText") {
    console.log("Background received translation request for:", req.text);
    
    fetch("http://127.0.0.1:5000/translate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text: req.text })
    })
    .then(async res => {
      console.log("Server responded with status:", res.status);
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Server error (${res.status}): ${errorText}`);
      }
      return res.json();
    })
    .then(data => {
      console.log("Success data:", data);
      sendResponse(data);
    })
    .catch(error => {
      console.error("Fetch failed completely:", error);
      sendResponse({ error: `Connection failed: ${error.message}. Make sure server is at 127.0.0.1:5000` });
    });
    return true; // Keep message channel open for async response
  }
});