// 더블클릭 이벤트 리스너 추가
document.addEventListener('dblclick', function() {
    const selectedText = window.getSelection().toString().trim();
    
    if (selectedText.length > 0) {
        console.log("선택된 텍스트:", selectedText);
        
        // 백그라운드 스크립트로 번역 요청 전송
        chrome.runtime.sendMessage({
            action: "translateText",
            text: selectedText
        }, (res) => {
            if (chrome.runtime.lastError) {
                console.error("메시지 전송 실패:", chrome.runtime.lastError);
                return;
            }
            
            if (res && res.error) {
                alert("에러 발생: " + res.error);
            } else if (res && res.translation) {
                alert("번역 내용 (" + res.text + "):\n" + res.translation);
            } else {
                alert("번역 결과를 가져올 수 없습니다.");
            }
        });
    }
});

// 기존 드래그 기능 메시지 수신 (필요 시 유지, 여기서는 주석 처리하거나 제거 가능)
chrome.runtime.onMessage.addListener((req)=>{
     if(req.action === 'startDrag'){
        alert("이제 단어를 더블클릭하면 바로 번역됩니다!");
     }
});