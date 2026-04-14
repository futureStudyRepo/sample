# 실행 명령어 터미널에서: uvicorn trans_fastapi:app --host 0.0.0.0 --port 5000 --reload
# 또는 파이썬으로 직접 실행: python trans_fastapi.py
# (사전 설치 필요: pip install fastapi uvicorn)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from googletrans import Translator

app = FastAPI()

# CORS 방지 정책 설정 (모든 도메인 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 클라이언트로부터 받을 JSON 데이터 구조
class TranslationRequest(BaseModel):
    text: str

@app.post("/translate")
async def translate(req: TranslationRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    try:
        # FastAPI는 비동기 처리(async/await)를 완벽히 네이티브로 지원하므로 코드가 아주 깔끔해집니다.
        translator = Translator()
        translation_result = await translator.translate(req.text, dest='en')
        
        return {
            "text": req.text,
            "translation": translation_result.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 코드 내에서 직접 uvicorn 호출 (debug=True 효과인 reload 옵션 포함)
    uvicorn.run("trans_fastapi:app", host="0.0.0.0", port=5000, reload=True)
