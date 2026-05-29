import React from "react";
import ReactDOM from "react-dom/client";
import { Visualizer3D } from "./panels/Visualizer3D";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Visualizer3D />
  </React.StrictMode>
);
