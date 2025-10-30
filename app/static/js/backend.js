async function sendMessage() {
  const query = document.getElementById("query").value.trim();
  if (!query) return;
  // 사용자 메시지 추가
  const userMsg = document.createElement("div");
  userMsg.className = "message user";
  userMsg.innerText = "🙋 " + query;
  document.getElementById("messages").appendChild(userMsg);
  // API 호출
  const response = await fetch("http://127.0.0.1:8001/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query })
  });
  const data = await response.json();
  // 챗봇 응답
  const botMsg = document.createElement("div");
  botMsg.className = "message bot";
  let text = "🤖 " + (data.generated_answer || "관련 규정을 찾을 수 없습니다.") + "\n\n";
  if (data.matches && data.matches.length > 0) {
    text += ":압정: 근거 문서:\n";
    data.matches.forEach((m, i) => {
      text += `[${i+1}] ${m.title} (최종점수: ${m.최종점수}, 의미점수: ${m.의미점수})\n`;
      text += m.text_snippet + "\n\n";
    });
  }
  const pre = document.createElement("pre");
  pre.innerText = text;
  botMsg.appendChild(pre);
  document.getElementById("messages").appendChild(botMsg);
  document.getElementById("query").value = "";
}
async function generateReport() {
  const box = document.getElementById("previewBox");
  const form = document.getElementById("projectForm");

  const project_name = document.getElementById("project_name").value.trim();
  const depart_name  = document.getElementById("depart_name").value.trim();
  const project_no   = document.getElementById("project_no").value.trim();
  const period       = document.getElementById("period").value.trim();
  const budget       = document.getElementById("budget").value.trim();
  const fileInput    = document.getElementById("attachment");

  if (!project_name || !depart_name || !project_no || !period || !budget) {
    alert("모든 항목을 입력해주세요.");
    return;
  }

  box.innerHTML = " 문서를 생성 중입니다... (환경에 따라 약5분 이상 소요됩니다.)";


  try {
    const formData = new FormData();
    formData.append("project_name", project_name);
    formData.append("depart_name", depart_name);
    formData.append("project_no", project_no);
    formData.append("period", period);
    formData.append("budget", budget);
    if (fileInput.files.length > 0) {
      formData.append("file", fileInput.files[0]);
    }
    const response = await fetch("http://127.0.0.1:8001/document-generator/generate_with_file", {
      method: "POST",
      body: formData
    });

    const data = await response.json();
    if (data.status === "success") {
      box.innerHTML = `
        <h2> 문서 생성 완료</h2>
        <p> ${data.message}</p>
        <p> 저장 경로: ${data.file_path}</p>
        <p><button onclick="downloadReport()"> 보고서 다운로드</button></p>
        <iframe style="position: relative; left: 10%; top: 10%; overflow-y: auto;" src="http://127.0.0.1:8001/document-generator/doc_page" width="700px" height="1000px"></iframe>
      `;
    } else {
      box.innerHTML = `<p style="color:red;">오류: ${data.error || "생성 실패"}</p>`;
    }
  } catch (err) {
    box.innerHTML = `<p style="color:red;">서버 연결 실패: ${err.message}</p>`;
  }
}

async function downloadReport() {
  const url = "http://127.0.0.1:8001/document-generator/download";
  const link = document.createElement("a");
  link.href = url;
  link.download = "RND_Report.docx";
  link.click();
}

let index = 0;
const slides = document.querySelectorAll('.carousel-item');
const total = slides.length;

function showSlide(i) {
  const inner = document.getElementById('carousel-inner');
  if (i >= total) index = 0;
  else if (i < 0) index = total - 1;
  else index = i;
  inner.style.transform = `translateX(-${index * 100}%)`;
}

function nextSlide() {
  showSlide(index + 1);
}

function prevSlide() {
  showSlide(index - 1);
}

document.getElementById("loadImageBtn").addEventListener("click", () => {
  const iframe = document.getElementById("myFrame");
  const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
  const img = iframeDoc.querySelector("img");

  if (img) {
    const newImg = document.createElement("img");
    newImg.src = img.src;
    newImg.alt = img.alt || "iframe image";
    newImg.style.maxWidth = "300px";
    document.getElementById("imageContainer").innerHTML = ""; // 기존 이미지 제거
    document.getElementById("imageContainer").appendChild(newImg);
  } else {
    alert("iframe 안에 이미지를 찾을 수 없습니다!");
  }
});
