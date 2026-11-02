"use client";

import Image from "next/image";
import { ChangeEvent, DragEvent, useState } from "react";

const models = [
  { id: "pretrained", label: "Best model" },
  { id: "fine-tuned", label: "Fine-tuned checkpoint" },
];

export default function Home() {
  const [model, setModel] = useState("pretrained");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [prediction, setPrediction] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function transcribe(nextFile: File, selectedModel: string) {
    setBusy(true);
    setError("");
    try {
      const body = new FormData();
      body.append("image", nextFile);
      body.append("model", selectedModel);
      const response = await fetch("/api/predict", { method: "POST", body });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Prediction failed.");
      setPrediction(result.prediction);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Prediction failed.");
    } finally {
      setBusy(false);
    }
  }

  function chooseFile(nextFile?: File) {
    if (!nextFile || !nextFile.type.startsWith("image/")) {
      setError("Choose an image file.");
      return;
    }
    setFile(nextFile);
    setPreview(URL.createObjectURL(nextFile));
    setPrediction("");
    void transcribe(nextFile, model);
  }

  function onInput(event: ChangeEvent<HTMLInputElement>) {
    chooseFile(event.target.files?.[0]);
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    chooseFile(event.dataTransfer.files[0]);
  }

  function changeModel(event: ChangeEvent<HTMLSelectElement>) {
    const nextModel = event.target.value;
    setModel(nextModel);
    if (file) void transcribe(file, nextModel);
  }

  return (
    <main className="page-shell">
      <header className="page-heading">
        <h1>Handwritten Text Recognition</h1>
        <p>Vision Transformer · TrOCR</p>
      </header>
      <section className="workspace">
        <select className="model-select" value={model} onChange={changeModel} aria-label="Choose model">
          <option value="" disabled>Choose Model</option>
          {models.map((option) => <option value={option.id} key={option.id}>{option.label}</option>)}
        </select>

        <label className="dropzone" onDragOver={(event) => event.preventDefault()} onDrop={onDrop}>
          <input type="file" accept="image/*" onChange={onInput} />
          {preview ? (
            <Image src={preview} alt="Selected manuscript" width={720} height={390} unoptimized />
          ) : (
            <svg className="upload-icon" viewBox="0 0 48 48" aria-hidden="true">
              <path d="M24 7v24m0-24 9 9m-9-9-9 9M9 29v9a3 3 0 0 0 3 3h24a3 3 0 0 0 3-3v-9" />
            </svg>
          )}
        </label>

        {file && (
          <button className="remove-button" onClick={() => {
            setFile(null);
            setPreview("");
            setPrediction("");
          }}>
            Remove image
          </button>
        )}

        {error && <p className="error">{error}</p>}
        <div className={`result-box ${prediction ? "ready" : ""}`}>
          {prediction ? <p>{prediction}</p> : <span>{busy ? "Reading…" : "Transcription"}</span>}
        </div>
      </section>
    </main>
  );
}
