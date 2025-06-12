"use client"

import { useState, useRef, useEffect } from "react"
import axios from "axios"
import "./App.css"

function App() {
  const [inputMessage, setInputMessage] = useState("")
  const [chatHistory, setChatHistory] = useState([])
  const [loading, setLoading] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [currentEmotion, setCurrentEmotion] = useState(null)
  const chatContainerRef = useRef(null)

  const startRecording = () => {
    setIsRecording(true)
    // In a real implementation, this would start the facial recognition
    // For now, we'll just set the state to show the video placeholder
  }

  const sendMessage = async (e) => {
    e?.preventDefault()

    if (!inputMessage.trim()) return

    // Simulate capturing an emotion (in real app, this would come from facial recognition)
    const emotions = ["neutral", "happy", "sad", "angry", "surprised", "confused"]
    const capturedEmotion = emotions[Math.floor(Math.random() * emotions.length)]
    setCurrentEmotion(capturedEmotion)

    // Add user message to chat
    const userMessage = {
      type: "user",
      content: inputMessage,
      emotion: capturedEmotion, // In real app, this would be the detected emotion
    }
    setChatHistory((prev) => [...prev, userMessage])

    // Clear input
    setInputMessage("")
    setLoading(true)

    try {
      // Call Python backend
      const res = await axios.post("http://127.0.0.1:5000/api/run-script", {
        data: inputMessage,
        emotion: capturedEmotion, // Send emotion to backend
      })

      // Simulate AI responding with an emotion (in real app, this would be determined by the backend)
      const aiEmotions = ["neutral", "happy", "sad", "concerned", "surprised", "confused"]
      const aiEmotion = aiEmotions[Math.floor(Math.random() * aiEmotions.length)]

      // Add bot response to chat
      const botMessage = {
        type: "bot",
        content: res.data.message || "No response from server",
        data: res.data,
        emotion: aiEmotion,
      }
      setChatHistory((prev) => [...prev, botMessage])

      // Update the displayed emotion for the AI
      setCurrentEmotion(aiEmotion)
    } catch (err) {
      console.error("Error calling API:", err)
      // Add error message to chat
      const errorMessage = {
        type: "error",
        content: "Failed to connect to the Python backend. Make sure the server is running.",
      }
      setChatHistory((prev) => [...prev, errorMessage])
      setCurrentEmotion("confused")
    } finally {
      setLoading(false)
    }
  }

  // Auto-scroll to bottom of chat when new messages arrive
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight
    }
  }, [chatHistory])

  return (
    <div className="app-container">
      <div className="main-content">
        <header className="app-header">
          <h1>Emotion-Aware AI Assistant</h1>
        </header>

        <div className="emotion-display-area">
          {!isRecording ? (
            <div className="start-screen">
              <div className="start-icon">🤖</div>
              <h2>Welcome to Emotion-Aware AI</h2>
              <p>Press the button below to start facial recognition</p>
              <button className="start-button" onClick={startRecording}>
                Start Facial Recognition
              </button>
            </div>
          ) : (
            <div className="video-emotion-container">
              {currentEmotion ? (
                <div className={`ai-face ${currentEmotion}`}>
                  {currentEmotion === "happy" && "😊"}
                  {currentEmotion === "sad" && "😢"}
                  {currentEmotion === "angry" && "😠"}
                  {currentEmotion === "surprised" && "😲"}
                  {currentEmotion === "confused" && "😕"}
                  {currentEmotion === "concerned" && "🙁"}
                  {currentEmotion === "neutral" && "😐"}
                  <div className="emotion-label">{currentEmotion}</div>
                </div>
              ) : (
                <div className="video-placeholder">
                  <div className="video-icon">📹</div>
                  <p>Video feed will appear here</p>
                  <p className="small-text">(Facial recognition active)</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="chat-sidebar">
        <div className="chat-header">
          <h2>Chat Assistant</h2>
        </div>

        <div className="chat-messages" ref={chatContainerRef}>
          {chatHistory.length === 0 ? (
            <div className="empty-chat">
              <p>Start facial recognition and send a message</p>
            </div>
          ) : (
            chatHistory.map((message, index) => (
              <div key={index} className={`message ${message.type}-message`}>
                {message.type === "user" && message.emotion && (
                  <div className="message-emotion user-emotion">
                    {message.emotion === "happy" && "😊"}
                    {message.emotion === "sad" && "😢"}
                    {message.emotion === "angry" && "😠"}
                    {message.emotion === "surprised" && "😲"}
                    {message.emotion === "confused" && "😕"}
                    {message.emotion === "neutral" && "😐"}
                  </div>
                )}
                <div className="message-content">
                  <p>{message.content}</p>
                  {message.type === "bot" && message.data && message.data !== message.content && (
                    <div className="message-data">
                      <details>
                        <summary>View full response</summary>
                        <pre>{JSON.stringify(message.data, null, 2)}</pre>
                      </details>
                    </div>
                  )}
                </div>
                {message.type === "bot" && message.emotion && (
                  <div className="message-emotion bot-emotion">
                    {message.emotion === "happy" && "😊"}
                    {message.emotion === "sad" && "😢"}
                    {message.emotion === "angry" && "😠"}
                    {message.emotion === "surprised" && "😲"}
                    {message.emotion === "confused" && "😕"}
                    {message.emotion === "concerned" && "🙁"}
                    {message.emotion === "neutral" && "😐"}
                  </div>
                )}
              </div>
            ))
          )}

          {loading && (
            <div className="message bot-message loading-message">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
        </div>

        <form className="chat-input-container" onSubmit={sendMessage}>
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Type your message here..."
            disabled={loading || !isRecording}
          />
          <button
            type="submit"
            disabled={loading || !inputMessage.trim() || !isRecording}
            className="send-button"
            aria-label="Send message"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </form>
      </div>
    </div>
  )
}

export default App

