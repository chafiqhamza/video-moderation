import React from 'react';
import { Bar } from 'react-chartjs-2';
import { Chart, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';

Chart.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function DashboardChart({ safeCount, totalFrames }) {
  const complianceRate = totalFrames ? ((safeCount / totalFrames) * 100).toFixed(1) : 0;
  const data = {
    labels: ['Safe Frames', 'Unsafe Frames'],
    datasets: [
      {
        label: 'Frame Count',
        data: [safeCount, totalFrames - safeCount],
        backgroundColor: ['#4caf50', '#f44336'],
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { position: 'top' },
      title: { display: true, text: `Compliance Rate: ${complianceRate}%` },
    },
  };

  return (
    <div style={{ maxWidth: 400, margin: '0 auto' }}>
      <Bar data={data} options={options} />
    </div>
  );
}

export default DashboardChart;
