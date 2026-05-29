import React from "react";
import ReactDOM from "react-dom/client";
import { WorkflowPanel } from "./panels/WorkflowPanel";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <WorkflowPanel />
  </React.StrictMode>
);
