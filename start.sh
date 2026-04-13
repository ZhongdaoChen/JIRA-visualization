#!/bin/bash
# 一键启动脚本：Flask 图表服务 + Streamlit 主应用

cd "$(dirname "$0")"

echo "▶ 启动 Flask 图表服务 (port 5050)..."
python3 chart_server.py > /tmp/flask_chart.log 2>&1 &
FLASK_PID=$!

# 等待 Flask 就绪
for i in {1..10}; do
  if curl -s http://127.0.0.1:5050/health > /dev/null 2>&1; then
    echo "✅ Flask 图表服务已就绪 (PID $FLASK_PID)"
    break
  fi
  sleep 1
done

echo "▶ 启动 Streamlit 主应用..."
streamlit run app.py

# Streamlit 退出后同时关闭 Flask
kill $FLASK_PID 2>/dev/null
echo "👋 所有服务已停止"
