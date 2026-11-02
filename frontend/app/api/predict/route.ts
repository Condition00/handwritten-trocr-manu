import { randomUUID } from "node:crypto";
import { writeFile, rm } from "node:fs/promises";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { NextResponse } from "next/server";

const run = promisify(execFile);
const projectRoot = path.resolve(process.cwd(), "..");
const models: Record<string, string> = {
  pretrained: "microsoft/trocr-small-handwritten",
  "fine-tuned": "outputs/fine_tune_lr1e5/epoch-2",
};

export async function POST(request: Request) {
  const form = await request.formData();
  const image = form.get("image");
  const modelKey = String(form.get("model") || "pretrained");
  const model = models[modelKey];
  if (!(image instanceof File) || !model) {
    return NextResponse.json({ error: "Image and valid model are required." }, { status: 400 });
  }
  if (!image.type.startsWith("image/")) {
    return NextResponse.json({ error: "Uploaded file must be an image." }, { status: 400 });
  }
  const tempPath = path.join("/tmp", `trocr-${randomUUID()}-${image.name.replace(/[^a-zA-Z0-9._-]/g, "_")}`);
  try {
    await writeFile(tempPath, Buffer.from(await image.arrayBuffer()));
    const { stdout } = await run("uv", ["run", "--project", projectRoot, "python", "infer_trocr.py", tempPath, "--model", model], {
      cwd: projectRoot,
      env: { ...process.env, HF_HOME: process.env.HF_HOME || "/tmp/dl-project-hf", HF_HUB_OFFLINE: "1" },
      maxBuffer: 1024 * 1024,
    });
    const lines = stdout.trim().split(/\r?\n/);
    return NextResponse.json({ prediction: lines.at(-1) || "" });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Model inference failed.";
    return NextResponse.json({ error: message.slice(-500) }, { status: 500 });
  } finally {
    await rm(tempPath, { force: true });
  }
}
