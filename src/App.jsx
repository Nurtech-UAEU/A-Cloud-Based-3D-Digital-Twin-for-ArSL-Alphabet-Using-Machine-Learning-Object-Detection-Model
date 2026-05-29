import React, { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import Loader from "./components/loader";
import ButtonHandler from "./components/btn-handler";
import { detectImage, detectVideo } from "./utils/detect";
import "./style/App.css";

const App = () => {
  const [loading, setLoading] = useState({ loading: true, progress: 0 });
  const [model, setModel] = useState({
    net: null,
    inputShape: [1, 0, 0, 3],
  });
  const [detectClass, setDetectClass] = useState("Idle");

  const imageRef = useRef(null);
  const cameraRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const modelName = "yolov5n";
  const classThreshold = 0.5;

  useEffect(() => {
    tf.ready().then(async () => {
      const yolov5 = await tf.loadGraphModel(
        `${window.location.href}/${modelName}_web_model/model.json`,
        {
          onProgress: (fractions) => {
            setLoading({ loading: true, progress: fractions });
          },
        }
      );

      const dummyInput = tf.ones(yolov5.inputs[0].shape);
      const warmupResult = await yolov5.executeAsync(dummyInput);
      tf.dispose(warmupResult);
      tf.dispose(dummyInput);

      setLoading({ loading: false, progress: 1 });
      setModel({
        net: yolov5,
        inputShape: yolov5.inputs[0].shape,
      });
    });
  }, []);

  const handleDetectClassChange = () => {
    const currentDetectClass = document.getElementById("detectclass").textContent;
    setDetectClass(currentDetectClass);
  };

  useEffect(() => {
    const intervalId = setInterval(handleDetectClassChange, 100);
    return () => clearInterval(intervalId);
  }, []);

  return (
    <div className="App">
      {loading.loading && (
        <Loader>
          Loading model... {(loading.progress * 100).toFixed(2)}%
        </Loader>
      )}
      <div className="header">
        <h1>Arabic Sign Language Alphabet 3D Digital Twin</h1>
        <div align="center">
          <model-viewer
            id="my3dmodel"
            autoplay
            animation-name={detectClass}
            src="moe.glb"
            camera-controls
          ></model-viewer>
        </div>
        <h1 id="detectclass">Idle</h1>
        <p>
          Serving : <code className="code">{modelName}</code>
        </p>
      </div>

      <div className="content">
        <img
          src="#"
          ref={imageRef}
          onLoad={() =>
            detectImage(imageRef.current, model, classThreshold, canvasRef.current)
          }
        />
        <video
          autoPlay
          muted
          ref={cameraRef}
          onPlay={() =>
            detectVideo(cameraRef.current, model, classThreshold, canvasRef.current)
          }
        />
        <video
          autoPlay
          muted
          ref={videoRef}
          onPlay={() =>
            detectVideo(videoRef.current, model, classThreshold, canvasRef.current)
          }
        />
        <canvas
          width={model.inputShape[1]}
          height={model.inputShape[2]}
          ref={canvasRef}
        />
      </div>

      <ButtonHandler imageRef={imageRef} cameraRef={cameraRef} videoRef={videoRef} />
    </div>
  );
};

export default App;

