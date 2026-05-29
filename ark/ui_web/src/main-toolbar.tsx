import React from "react";
import ReactDOM from "react-dom/client";
import { ToolbarPanel } from "./panels/ToolbarPanel";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ToolbarPanel />
  </React.StrictMode>
);
