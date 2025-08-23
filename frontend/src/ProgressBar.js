import React, { useEffect, useState } from "react";
import './ProgressBar.css';

export default function ProgressBar() {
  const [current, setCurrent] = useState(0);
  const [total, setTotal] = useState(0);
  const [active, setActive] = useState(false);

  useEffect(() => {
    const ws = new window.WebSocket("ws://localhost:8000/ws/progress");
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "progress") {
        setCurrent(data.current);
        setTotal(data.total);
        setActive(data.active);
      }
    };
    ws.onclose = () => setActive(false);
    return () => ws.close();
  }, []);

  const percent = total > 0 ? Math.round((current / total) * 100) : 0;

  return (
    <div className="progress-container">
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: percent + "%" }}></div>
      </div>
      <div className="progress-text">
        {active
          ? `Analyse des frames : ${current} / ${total} (${percent}%)`
          : total > 0
          ? "Analyse terminée !"
          : "En attente d'analyse..."}
      </div>
    </div>
  );
}
