import "./App.css";
import Sol from "./components/solar-system";
import { Canvas } from "react-three-fiber";

function App() {
  return (
    <div className="App">
      <Canvas>
        <Sol />
      </Canvas>
    </div>
  );
}

export default App;
