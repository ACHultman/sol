import Sol from "./components/solar-system";
import { Canvas } from "react-three-fiber";
import React from "react";
import "./app.css";

function App() {
  return (
    <div className="app">
      <Canvas style={{ backgroundColor: "black" }} camera={[40, 0, 0]}>
        <Sol />
      </Canvas>
    </div>
  );
}

export default App;
