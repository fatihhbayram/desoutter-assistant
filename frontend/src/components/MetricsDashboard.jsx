import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Dashboard.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://192.168.1.125:8000';

const MetricsDashboard = ({ token }) => {
  const [metricsHealth, setMetricsHealth] = useState(null);
  const [metricsStats, setMetricsStats] = useState(null);
  const [recentQueries, setRecentQueries] = useState([]);
  const [slowQueries, setSlowQueries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [error, setError] = useState(null);

  // Load all metrics on mount
  useEffect(() => {
    loadMetrics();
  }, []);

  // Auto-refresh health every 30s
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      loadHealth();
    }, 30000);
    
    return () => clearInterval(interval);
  }, [autoRefresh]);

  const loadMetrics = async () => {
    setLoading(true);
    try {
      await Promise.all([
        loadHealth(),
        loadStats(),
        loadQueries(),
        loadSlowQueries()
      ]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/admin/metrics/health`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setMetricsHealth(response.data);
    } catch (err) {
      console.error('Error loading health:', err);
    }
  };

  const loadStats = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/admin/metrics/stats`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setMetricsStats(response.data);
    } catch (err) {
      console.error('Error loading stats:', err);
    }
  };

  const loadQueries = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/admin/metrics/queries`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setRecentQueries(response.data.queries || []);
    } catch (err) {
      console.error('Error loading queries:', err);
    }
  };

  const loadSlowQueries = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/admin/metrics/slow`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setSlowQueries(response.data.slow_queries || []);
    } catch (err) {
      console.error('Error loading slow queries:', err);
    }
  };

  const handleReset = async () => {
    if (!window.confirm('Reset all metrics? This cannot be undone.')) return;
    
    try {
      await axios.post(`${API_BASE_URL}/admin/metrics/reset`, {}, {
        headers: { Authorization: `Bearer ${token}` }
      });
      alert('Metrics reset successfully');
      loadMetrics();
    } catch (err) {
      alert(`Error: ${err.message}`);
    }
  };

  if (loading) {
    return (
      <div className="loading">
        <div className="spinner"></div>
        <p>Loading metrics...</p>
      </div>
    );
  }

  if (error) {
    return <div className="error">Error: {error}</div>;
  }

  return (
    <div className="metrics-dashboard">
      <div className="dashboard-header">
        <h2>⚡ Performance Metrics</h2>
        <div className="dashboard-actions">
          <label>
            <input 
              type="checkbox" 
              checked={autoRefresh} 
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh (30s)
          </label>
          <button onClick={loadMetrics} className="btn-primary">🔄 Refresh</button>
          <button onClick={handleReset} className="btn-danger">Reset Metrics</button>
        </div>
      </div>

      {/* System Health Card */}
      <div className="metrics-card">
        <h3>🏥 System Health</h3>
        {metricsHealth && (
          <div className="health-status">
            <div className={`status-indicator ${metricsHealth.status || 'healthy'}`}>
              {metricsHealth.status || 'healthy'}
            </div>
            <div className="health-details">
              <p><strong>Uptime:</strong> {metricsHealth.uptime || 'N/A'}</p>
              <p><strong>Last Check:</strong> {metricsHealth.timestamp ? new Date(metricsHealth.timestamp).toLocaleString() : 'N/A'}</p>
            </div>
          </div>
        )}
      </div>

      {/* Performance Stats Card */}
      <div className="metrics-card">
        <h3>📊 Performance Statistics</h3>
        {metricsStats && (
          <div className="stats-grid">
            <div className="stat">
              <span className="stat-label">Total Queries</span>
              <span className="stat-value">{metricsStats.total_queries || 0}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Avg Response Time</span>
              <span className="stat-value">{metricsStats.avg_response_time || 0}ms</span>
            </div>
            <div className="stat">
              <span className="stat-label">Cache Hit Rate</span>
              <span className="stat-value">{metricsStats.cache_hit_rate || 0}%</span>
            </div>
            <div className="stat">
              <span className="stat-label">Slow Queries</span>
              <span className="stat-value">{metricsStats.slow_query_count || 0}</span>
            </div>
          </div>
        )}
      </div>

      {/* Recent Queries Table */}
      <div className="metrics-card">
        <h3>🔍 Recent Queries (Last 20)</h3>
        {recentQueries.length > 0 ? (
          <table className="query-table">
            <thead>
              <tr>
                <th>Query</th>
                <th>Response Time</th>
                <th>Cache</th>
                <th>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {recentQueries.slice(0, 20).map((q, idx) => (
                <tr key={idx}>
                  <td>{q.query ? q.query.substring(0, 50) + '...' : 'N/A'}</td>
                  <td>{q.response_time || 0}ms</td>
                  <td className={q.cache_hit ? 'cache-hit' : 'cache-miss'}>
                    {q.cache_hit ? 'HIT' : 'MISS'}
                  </td>
                  <td>{q.timestamp ? new Date(q.timestamp).toLocaleString() : 'N/A'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p>No recent queries</p>
        )}
      </div>

      {/* Slow Queries Alert */}
      {slowQueries.length > 0 && (
        <div className="metrics-card alert">
          <h3>⚠️ Slow Queries (&gt;10s)</h3>
          <table className="query-table">
            <thead>
              <tr>
                <th>Query</th>
                <th>Response Time</th>
                <th>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {slowQueries.map((q, idx) => (
                <tr key={idx}>
                  <td>{q.query ? q.query.substring(0, 50) + '...' : 'N/A'}</td>
                  <td className="slow">{q.response_time || 0}ms</td>
                  <td>{q.timestamp ? new Date(q.timestamp).toLocaleString() : 'N/A'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default MetricsDashboard;
