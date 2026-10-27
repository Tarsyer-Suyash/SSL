import argparse
import ncnn
import os
import zmq
import numpy as np
import cv2
import msgpack
import time


class NCNNRunner:
    def __init__(self, model_path, use_gpu=False):
        self.model_path = model_path
        self.net = ncnn.Net()
        param_path = os.path.join(model_path, "model.ncnn.param")
        self.net.load_param(param_path)
        model_bin = os.path.join(model_path, "model.ncnn.bin")
        self.net.load_model(model_bin)
        print(f"Model loaded from: {model_path}")

    def preprocess(self, im):
        img = im[..., ::-1].transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255.0
        return img

    def predict(self, img2):
        mat_in = ncnn.Mat(img2[0])
        with self.net.create_extractor() as ex:
            ex.input(self.net.input_names()[0], mat_in)
            y = [np.array(ex.extract(x)[1])[None]
                 for x in sorted(self.net.output_names())]
        return y

    def postprocess(self, output, confidence_thres, iou_thres):
        outputs = np.transpose(np.squeeze(output[0]))
        rows    = outputs.shape[0]
        boxes, scores, class_ids = [], [], []

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score      = np.amax(classes_scores)
            if max_score >= confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                left   = int(x - w / 2)
                top    = int(y - h / 2)
                right  = int(x + w / 2)
                bottom = int(y + h / 2)
                boxes.append([left, top, right, bottom])
                scores.append(max_score)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)
        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([])

        indices         = indices.flatten()
        final_boxes     = np.array(boxes)[indices]
        final_scores    = np.array(scores)[indices]
        final_class_ids = np.array(class_ids)[indices]
        return final_boxes, final_scores, final_class_ids

    def run(self, input_image, confidence_thresh, iou_thres):
        im2 = self.preprocess(input_image)
        y   = self.predict(im2)
        return self.postprocess(y, confidence_thresh, iou_thres)


# ─── Argument parser ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="NCNN detector ZMQ server")
parser.add_argument('--model_path',        type=str,   required=True)
parser.add_argument('--port',              type=int,   default=5555)   # ← NEW
parser.add_argument('--confidence_thresh', type=float, default=0.2)
parser.add_argument('--iou_thres',         type=float, default=0.45)
args = parser.parse_args()

# ─── ZMQ setup ────────────────────────────────────────────────────────────────
context  = zmq.Context()
receiver = context.socket(zmq.ROUTER)
receiver.bind(f"tcp://127.0.0.1:{args.port}")
print(f"ZMQ server listening on port {args.port}")

# ─── Load model ───────────────────────────────────────────────────────────────
yolo_ncnn = NCNNRunner(
    model_path=f'/home/celestial/detector/models/{args.model_path}'
)

print("Ready for inference...")

# ─── Main loop ────────────────────────────────────────────────────────────────
while True:
    sender_identity, _, image_data = receiver.recv_multipart()
    start_time = time.time()

    image_np = np.frombuffer(image_data, dtype=np.uint8)
    frame    = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    bbox_lists, scores, class_lists = yolo_ncnn.run(
        input_image=frame,
        confidence_thresh=args.confidence_thresh,
        iou_thres=args.iou_thres
    )

    message = {
        'bboxes':  bbox_lists.astype(int).tolist() if len(bbox_lists) else [],
        'classes': class_lists.astype(int).tolist() if len(class_lists) else [],
        'scores':  scores.tolist() if len(scores) else []
    }
    print(f"Port {args.port} | time: {time.time()-start_time:.3f}s | "
          f"detections: {len(message['bboxes'])}")
    receiver.send_multipart([sender_identity, msgpack.packb(message)])
