import React from "react";
import ReactDOM from "react-dom/client";
import { ArchitectPanel } from "./panels/ArchitectPanel";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ArchitectPanel />
  </React.StrictMode>
);
