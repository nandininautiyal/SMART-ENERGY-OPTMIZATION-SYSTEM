import React, { useEffect, useState, useCallback } from "react";
import axios from "axios";
import { Bar } from "react-chartjs-2";
import "chart.js/auto";

import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  TextField
} from "@mui/material";

// ✅ IMPORTANT: Works for Docker + local
const API = process.env.REACT_APP_API || "http://localhost:8000";

function App() {
  const [rooms, setRooms] = useState({
    roomA: "",
    roomB: "",
    roomC: ""
  });

  const [results, setResults] = useState([]);
  const [chartData, setChartData] = useState(null);
  const [rImage, setRImage] = useState(null);

  // ---------------- FETCH R IMAGE ----------------
  const fetchR = useCallback(() => {
    setRImage(`${API}/static/plot.png?${new Date().getTime()}`);
  }, []);

  // ---------------- ANALYZE ROOMS ----------------
  const analyzeRooms = () => {
    axios.post(`${API}/analyze-rooms`, {
      rooms: [
        { name: "Room A", power: parseFloat(rooms.roomA) || 0 },
        { name: "Room B", power: parseFloat(rooms.roomB) || 0 },
        { name: "Room C", power: parseFloat(rooms.roomC) || 0 }
      ]
    })
    .then(res => {
      setResults(res.data.results);

      // Chart
      const labels = res.data.results.map(r => r.room);
      const values = res.data.results.map(r => r.power);

      setChartData({
        labels,
        datasets: [
          {
            label: "Power Usage",
            data: values,
            backgroundColor: "#42a5f5"
          }
        ]
      });
    })
    .catch(err => {
      console.error("Error:", err);
      alert("Backend not reachable 🚨");
    });
  };

  // ---------------- LOAD R GRAPH ----------------
  useEffect(() => {
    fetchR();
  }, [fetchR]);

  return (
    <Container maxWidth="lg" style={{ marginTop: "30px" }}>

      {/* HEADER */}
      <Card style={{
        background: "linear-gradient(45deg, #1976d2, #42a5f5)",
        color: "white",
        marginBottom: "20px"
      }}>
        <CardContent>
          <Typography variant="h4">
            ⚡ Smart Energy Optimization System
          </Typography>
          <Typography>
            AI-powered monitoring using Spark, TinyML, R & ML
          </Typography>
        </CardContent>
      </Card>

      {/* INPUT SECTION */}
      <Card style={{ marginBottom: "20px" }}>
        <CardContent>
          <Typography variant="h6">🏠 Room Energy Input</Typography>

          <Grid container spacing={2}>
            <Grid item xs={4}>
              <TextField
                label="Room A"
                fullWidth
                value={rooms.roomA}
                onChange={(e) =>
                  setRooms({ ...rooms, roomA: e.target.value })
                }
              />
            </Grid>

            <Grid item xs={4}>
              <TextField
                label="Room B"
                fullWidth
                value={rooms.roomB}
                onChange={(e) =>
                  setRooms({ ...rooms, roomB: e.target.value })
                }
              />
            </Grid>

            <Grid item xs={4}>
              <TextField
                label="Room C"
                fullWidth
                value={rooms.roomC}
                onChange={(e) =>
                  setRooms({ ...rooms, roomC: e.target.value })
                }
              />
            </Grid>
          </Grid>

          <Button
            variant="contained"
            style={{ marginTop: "15px" }}
            onClick={analyzeRooms}
          >
            ANALYZE ROOMS
          </Button>
        </CardContent>
      </Card>

      {/* RESULTS */}
      {results.map((r, i) => (
        <Card
          key={i}
          style={{
            marginBottom: "10px",
            background: r.anomaly ? "#ffebee" : "#e8f5e9"
          }}
        >
          <CardContent>
            <Typography variant="h6">{r.room}</Typography>
            <Typography>⚡ Power: {r.power}</Typography>
            <Typography>{r.decision}</Typography>
            <Typography>💡 {r.remedy}</Typography>
            {r.anomaly && <Typography>🚨 Anomaly detected</Typography>}
          </CardContent>
        </Card>
      ))}

      {/* CHART */}
      {chartData && (
        <Card style={{ marginTop: "20px" }}>
          <CardContent>
            <Bar data={chartData} />
          </CardContent>
        </Card>
      )}

      {/* R ANALYSIS */}
      <Card style={{ marginTop: "20px" }}>
        <CardContent>
          <Typography variant="h6">📊 R Statistical Analysis</Typography>
          {rImage ? (
            <img
              src={rImage}
              alt="R Plot"
              style={{ width: "100%" }}
            />
          ) : (
            <Typography>Loading R plot...</Typography>
          )}
        </CardContent>
      </Card>

    </Container>
  );
}

export default App;