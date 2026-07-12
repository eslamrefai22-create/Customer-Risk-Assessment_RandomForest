document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("assessmentForm");
  const submitBtn = document.getElementById("submitBtn");
  const formError = document.getElementById("formError");

  const verdictEmpty = document.getElementById("verdictEmpty");
  const verdictResult = document.getElementById("verdictResult");
  const stamp = document.getElementById("stamp");
  const stampPct = document.getElementById("stampPct");
  const stampWord = document.getElementById("stampWord");
  const verdictRisk = document.getElementById("verdictRisk");
  const stayPct = document.getElementById("stayPct");
  const churnPct = document.getElementById("churnPct");

  const fileDate = document.getElementById("fileDate");
  fileDate.textContent = new Date().toLocaleDateString(undefined, {
    year: "numeric", month: "short", day: "numeric",
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    formError.textContent = "";

    const payload = {
      CreditScore: document.getElementById("CreditScore").value,
      Geography: document.getElementById("Geography").value,
      Gender: document.getElementById("Gender").value,
      Age: document.getElementById("Age").value,
      Tenure: document.getElementById("Tenure").value,
      Balance: document.getElementById("Balance").value,
      NumOfProducts: document.getElementById("NumOfProducts").value,
      HasCrCard: document.getElementById("HasCrCard").checked ? 1 : 0,
      IsActiveMember: document.getElementById("IsActiveMember").checked ? 1 : 0,
      EstimatedSalary: document.getElementById("EstimatedSalary").value,
    };

    submitBtn.disabled = true;
    submitBtn.querySelector("span").textContent = "Assessing…";

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (!res.ok || !data.ok) {
        throw new Error(data.error || "Something went wrong. Please check the values and try again.");
      }

      renderVerdict(data);
    } catch (err) {
      formError.textContent = err.message;
    } finally {
      submitBtn.disabled = false;
      submitBtn.querySelector("span").textContent = "Assess risk";
    }
  });

  function renderVerdict(data) {
    verdictEmpty.style.display = "none";
    verdictResult.hidden = false;
    verdictResult.classList.remove("is-visible");
    void verdictResult.offsetWidth; // restart animation
    verdictResult.classList.add("is-visible");

    const riskClass =
      data.risk_level === "High" ? "stamp--high" :
      data.risk_level === "Medium" ? "stamp--med" : "stamp--low";

    stamp.className = `stamp ${riskClass}`;
    // restart the stamp-in animation
    stamp.style.animation = "none";
    void stamp.offsetWidth;
    stamp.style.animation = "";

    stampPct.textContent = `${data.churn_probability}%`;
    stampWord.textContent = data.will_churn ? "Churn risk" : "Likely to stay";
    verdictRisk.textContent = `${data.risk_level} risk of churn`;
    stayPct.textContent = `${data.stay_probability}%`;
    churnPct.textContent = `${data.churn_probability}%`;
  }
});
