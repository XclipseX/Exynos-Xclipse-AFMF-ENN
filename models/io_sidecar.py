"""Sinh <name>.e9965.io.json từ TFLite gốc (raw). API: tf.lite.Interpreter (hoặc ai_edge_litert.interpreter.Interpreter)."""
import json, sys, os
try:
    from ai_edge_litert.interpreter import Interpreter
except Exception:
    import tensorflow as tf; Interpreter = tf.lite.Interpreter
def sidecar(raw, out):
    it = Interpreter(model_path=raw); it.allocate_tensors()
    def d(x):
        sc, zp = x["quantization"]
        return {"name": x["name"], "shape": [int(v) for v in x["shape"]], "dtype": __import__("numpy").dtype(x["dtype"]).name,
                "scale": float(sc), "zero_point": int(zp)}
    j = {"inputs": [d(x) for x in it.get_input_details()], "outputs": [d(x) for x in it.get_output_details()]}
    json.dump(j, open(out, "w"), indent=1); return j
if __name__ == "__main__":
    raw = sys.argv[1]; name = os.path.basename(raw).replace("_raw.tflite", "")
    print(json.dumps(sidecar(raw, sys.argv[2] if len(sys.argv) > 2 else name + ".e9965.io.json"), indent=1))
