import React from "react";
import { Header } from "./components/Header";
import { ChatContainer } from "./components/ChatContainer";
import { InputBar } from "./components/InputBar";
import { useChat } from "./hooks/useChat";

export const App: React.FC = () => {
  const {
    messages,
    isLoading,
    sendMessage,
    stopStreaming,
    clearChat,
  } = useChat();

  return (
    <div className="app-container client-mode">
      <div className="main-content">
        <Header onNewChat={clearChat} />

        <ChatContainer
          messages={messages}
          isLoading={isLoading}
          onSelectQuery={sendMessage}
        />

        <InputBar
          onSendMessage={sendMessage}
          isLoading={isLoading}
          onStop={stopStreaming}
        />
      </div>
    </div>
  );

};

export default App;
