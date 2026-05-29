import React from "react";
import ReactDOM from "react-dom/client";
import { TerminalPanel } from "./panels/TerminalPanel";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <TerminalPanel />
  </React.StrictMode>
);
