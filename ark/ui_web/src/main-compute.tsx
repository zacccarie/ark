import React from "react";
import ReactDOM from "react-dom/client";
import { ComputePanel } from "./panels/ComputePanel";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ComputePanel />
  </React.StrictMode>
);
