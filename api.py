# 导入Flask库
from flask import Flask, jsonify, request
from flask_cors import CORS
from celery import Celery
from tasks import generate_image
import config

# 初始化应用程序
app = Flask(__name__)
CORS(app)
celery_app = Celery(config.celery_name, broker=config.BROKER_URL, backend=config.BACKEND_URL)

# 为API定义路由
@app.route('/api/image_genarate', methods=['post'])
def get_users():
    data = request.json
    prompt = data["prompt"]
    model = data["model"]
    if model == "" or model == "Realistic":
        model_id = "Realistic"
    elif model == "GhostMix":
        model_id = "GhostMix"
    task_result = generate_image.delay(prompt, model_id)
    print("****prompt****:", prompt)
    return jsonify({'success': 1, "task_id": task_result.id, "msg": "success"})


@app.route('/api/get_task_genarate', methods=['GET'])
def get_add():
    task_id = request.args.get('task_id')
    if not task_id:
        return jsonify({'success': 0, "reason": "loss task_id"})
    task_result = celery_app.AsyncResult(task_id)
    print("****task_result****:", task_result.status)
    if task_result.status == "PENDING":
        return jsonify({'success': 0, "reason": "任务队列中..."})
    elif task_result.status == "STARTED":
        return jsonify({'success': 0, "reason": "图片生成中..."})
    file_link = task_result.result
    if file_link == "ERR1":
        return {"success": 0, "reason": "翻译组件错误"}
    elif file_link == "ERR2":
        return {"success": 0, "reason": "模型选择错误"}
    msg = {"success": 1, "reason": "PASS", "file_link": file_link}
    return jsonify(msg)


# 启动应用程序
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)

