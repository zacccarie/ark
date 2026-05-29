import React from "react";
import ReactDOM from "react-dom/client";
import { InspectorPanel } from "./panels/InspectorPanel";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <InspectorPanel />
  </React.StrictMode>
);
