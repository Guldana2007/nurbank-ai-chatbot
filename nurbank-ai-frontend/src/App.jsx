import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:9000";

function App() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(false);

  async function sendMessage() {
    if (!question.trim()) return;

    const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    // добавляем сообщение пользователя
    setMessages((msgs) => [
      ...msgs,
      { sender: "user", text: question, time }
    ]);

    const currentQuestion = question;
    setQuestion("");
    setLoading(true);

    try {
      const resp = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: currentQuestion })
      });
      const data = await resp.json();

      const botTime = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

      setMessages((msgs) => [
        ...msgs,
        { sender: "bot", text: data.answer, time: botTime }
      ]);
    } catch (e) {
      const errTime = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

      setMessages((msgs) => [
        ...msgs,
        { sender: "bot", text: "⚠️ Ошибка: " + e.message, time: errTime }
      ]);
    }

    setLoading(false);
  }

  return (
    <div className={darkMode ? "dark" : "light"} style={{
      maxWidth: "600px",
      margin: "40px auto",
      fontFamily: "Arial, sans-serif"
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2>💬 NurBank AI Assistant</h2>
        <button onClick={() => setDarkMode(!darkMode)}>
          {darkMode ? "☀️" : "🌙"}
        </button>
      </div>

      <div style={{
        border: "1px solid #ccc",
        borderRadius: "8px",
        padding: "10px",
        minHeight: "300px",
        background: darkMode ? "#2c2c2c" : "#fafafa",
        color: darkMode ? "#f1f1f1" : "#000"
      }}>
        {messages.map((msg, i) => (
          <div key={i} style={{
            textAlign: msg.sender === "user" ? "right" : "left",
            margin: "8px 0"
          }}>
            <div style={{
              display: "inline-block",
              padding: "8px 12px",
              borderRadius: "6px",
              background: msg.sender === "user"
                ? (darkMode ? "#1d70a2" : "#d0ebff")
                : (darkMode ? "#444" : "#e9ecef"),
              color: msg.sender === "user" ? "#fff" : "inherit"
            }}>
              {msg.text}
              <div style={{ fontSize: "0.7em", opacity: 0.7, marginTop: "4px" }}>
                {msg.time}
              </div>
            </div>
          </div>
        ))}

        {loading && (
          <div style={{ fontStyle: "italic", opacity: 0.7 }}>Бот печатает…</div>
        )}
      </div>

      <div style={{ display: "flex", marginTop: "10px" }}>
        <input
          style={{ flex: 1, padding: "10px" }}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyPress={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Введите вопрос..."
        />
        <button
          onClick={sendMessage}
          style={{
            marginLeft: "10px",
            padding: "10px 15px",
            background: "#0066cc",
            color: "white",
            border: "none",
            borderRadius: "4px"
          }}
        >
          Отправить
        </button>
      </div>
    </div>
  );
}

export default App;
