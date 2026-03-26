import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './BruteForcePanel.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://192.168.1.125:8000';

const BruteForcePanel = () => {
  const [stats, setStats] = useState(null);
  const [blockedList, setBlockedList] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [unblockUsername, setUnblockUsername] = useState('');
  const [unblockIp, setUnblockIp] = useState('');
  const [message, setMessage] = useState(null);
  const [countdown, setCountdown] = useState({});
  const [refreshing, setRefreshing] = useState(false);

  const getAuthHeader = () => {
    const token = localStorage.getItem('auth_token');
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  const fetchStats = async () => {
    try {
      const response = await axios.get(
        `${API_BASE_URL}/admin/security/brute-force/stats`,
        { headers: getAuthHeader() }
      );
      setStats(response.data);
      setError(null);
    } catch (err) {
      console.error('Error fetching stats:', err);
      setError('Failed to load statistics');
    }
  };

  const fetchBlockedList = async () => {
    try {
      const response = await axios.get(
        `${API_BASE_URL}/admin/security/brute-force/blocked?limit=100`,
        { headers: getAuthHeader() }
      );
      setBlockedList(response.data.blocked || []);
      setLoading(false);
      setError(null);
    } catch (err) {
      console.error('Error fetching blocked list:', err);
      setError('Failed to load blocked list');
      setLoading(false);
    }
  };

  const refreshData = async () => {
    setRefreshing(true);
    await Promise.all([fetchStats(), fetchBlockedList()]);
    setTimeout(() => setRefreshing(false), 500);
  };

  useEffect(() => {
    fetchStats();
    fetchBlockedList();
    const interval = setInterval(() => {
      fetchStats();
      fetchBlockedList();
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const timer = setInterval(() => {
      const newCountdown = {};
      blockedList.forEach((item, index) => {
        const blockedUntil = new Date(item.blocked_until);
        const now = new Date();
        const diff = Math.max(0, Math.floor((blockedUntil - now) / 1000));
        if (diff > 0) {
          const minutes = Math.floor(diff / 60);
          const seconds = diff % 60;
          newCountdown[index] = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        } else {
          newCountdown[index] = 'Expired';
        }
      });
      setCountdown(newCountdown);
    }, 1000);
    return () => clearInterval(timer);
  }, [blockedList]);

  const handleUnblock = async (username = null, ip = null) => {
    try {
      const params = new URLSearchParams();
      if (username) params.append('username', username);
      if (ip) params.append('ip_address', ip);

      const response = await axios.delete(
        `${API_BASE_URL}/admin/security/brute-force/unblock?${params}`,
        { headers: getAuthHeader() }
      );

      setMessage({ type: 'success', text: response.data.message });
      fetchStats();
      fetchBlockedList();
      setUnblockUsername('');
      setUnblockIp('');
      setTimeout(() => setMessage(null), 3000);
    } catch (err) {
      setMessage({
        type: 'error',
        text: err.response?.data?.detail || 'Failed to unblock'
      });
      setTimeout(() => setMessage(null), 5000);
    }
  };

  const handleCleanup = async () => {
    if (!window.confirm('Delete records older than 30 days?')) return;

    try {
      const response = await axios.post(
        `${API_BASE_URL}/admin/security/brute-force/cleanup`,
        { days: 30 },
        { headers: getAuthHeader() }
      );

      setMessage({ type: 'success', text: response.data.message });
      fetchStats();
      fetchBlockedList();
      setTimeout(() => setMessage(null), 3000);
    } catch (err) {
      setMessage({
        type: 'error',
        text: err.response?.data?.detail || 'Cleanup failed'
      });
      setTimeout(() => setMessage(null), 5000);
    }
  };

  if (loading) {
    return (
      <div className="brute-force-panel">
        <div className="loading">
          <div className="spinner"></div>
          <p>Loading security data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="brute-force-panel">
      {/* Header */}
      <div className="dashboard-header">
        <div>
          <h2>🛡️ Brute Force Protection</h2>
          <p className="subtitle">Monitor and manage failed login attempts and blocked IP addresses</p>
        </div>
        <button
          onClick={refreshData}
          disabled={refreshing}
          className="btn-refresh"
        >
          <span className={refreshing ? 'spinning' : ''}>🔄</span>
          {refreshing ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {/* Alert Messages */}
      {error && (
        <div className="alert alert-error">
          <span className="alert-icon">⚠️</span>
          <div>
            <strong>Error</strong>
            <p>{error}</p>
          </div>
        </div>
      )}

      {message && (
        <div className={`alert ${message.type === 'success' ? 'alert-success' : 'alert-error'}`}>
          <span className="alert-icon">{message.type === 'success' ? '✅' : '❌'}</span>
          <p>{message.text}</p>
        </div>
      )}

      {/* Protection Status Banner */}
      {stats && (
        <div className={`protection-banner ${stats.enabled ? 'enabled' : 'disabled'}`}>
          <div className="banner-icon">🛡️</div>
          <div className="banner-content">
            <h3>All shield modules operational.</h3>
            <p>System is currently under active monitoring.</p>
          </div>
        </div>
      )}

      {/* Stats Grid */}
      {stats && (
        <div className="security-stats-grid">
          <div className="stat-card">
            <div className="stat-header">ACTIVE BANS</div>
            <div className="stat-body">
              <div className="stat-number">{stats.currently_blocked || 0}</div>
              <div className="stat-icon">🚫</div>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-header">RECENT ATTEMPTS</div>
            <div className="stat-body">
              <div className="stat-number">{stats.recent_attempts_24h || 0}</div>
              <div className="stat-icon">📊</div>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-header">LAST 24H</div>
            <div className="stat-body">
              <div className="stat-number">{stats.recent_attempts_24h || 0}</div>
              <div className="stat-icon">🕐</div>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-header">TOTAL RECORDS</div>
            <div className="stat-body">
              <div className="stat-number">{stats.total_records || 0}</div>
              <div className="stat-icon">💾</div>
            </div>
          </div>
        </div>
      )}

      {/* Ban Configurations */}
      {stats?.config && (
        <div className="metrics-card">
          <h3>⚙️ Ban Configurations</h3>
          <div className="ban-config-table">
            <table>
              <thead>
                <tr>
                  <th>LEVEL</th>
                  <th>TRIGGERS</th>
                  <th>DURATION</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><span className="level-badge level-1">Level 1</span></td>
                  <td>{stats.config.max_attempts_level_1} Attempts</td>
                  <td><span className="duration-badge">{stats.config.ban_duration_level_1} MIN</span></td>
                </tr>
                <tr>
                  <td><span className="level-badge level-2">Level 2</span></td>
                  <td>{stats.config.max_attempts_level_2} Attempts</td>
                  <td><span className="duration-badge">{stats.config.ban_duration_level_2 / 60} HOUR</span></td>
                </tr>
                <tr>
                  <td><span className="level-badge level-3">Level 3</span></td>
                  <td>{stats.config.max_attempts_level_3} Attempts</td>
                  <td><span className="duration-badge permanent">PERMANENT</span></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Blocked List */}
      <div className="metrics-card">
        <h3>🚫 Blocked Users & IP Addresses</h3>
        {blockedList.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">✅</div>
            <h4>All Clear!</h4>
            <p>No blocked entries at the moment</p>
          </div>
        ) : (
          <div className="blocked-table">
            <table>
              <thead>
                <tr>
                  <th>Username</th>
                  <th>IP Address</th>
                  <th>Attempts</th>
                  <th>Ban Level</th>
                  <th>Time Remaining</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {blockedList.map((item, index) => (
                  <tr key={index}>
                    <td>
                      <div className="user-cell">
                        <span className="user-icon">👤</span>
                        {item.username}
                      </div>
                    </td>
                    <td className="mono">{item.ip_address}</td>
                    <td>
                      <span className="attempts-badge">{item.attempts}</span>
                    </td>
                    <td>
                      <span className={`level-badge level-${item.ban_level}`}>
                        Level {item.ban_level}
                      </span>
                    </td>
                    <td className="mono countdown">
                      {countdown[index] || `${item.retry_after_minutes} min`}
                    </td>
                    <td>
                      <button
                        onClick={() => handleUnblock(item.username, item.ip_address)}
                        className="btn-unblock"
                      >
                        🔓 Unblock
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Manual Unblock */}
      <div className="metrics-card">
        <h3>🔓 Manual Unblock</h3>
        <div className="unblock-form">
          <div className="form-group">
            <label>Unblock by Username</label>
            <div className="input-group">
              <input
                type="text"
                value={unblockUsername}
                onChange={(e) => setUnblockUsername(e.target.value)}
                placeholder="Enter username"
              />
              <button
                onClick={() => handleUnblock(unblockUsername, null)}
                disabled={!unblockUsername}
                className="btn-primary"
              >
                Unblock
              </button>
            </div>
          </div>

          <div className="form-group">
            <label>Unblock by IP Address</label>
            <div className="input-group">
              <input
                type="text"
                value={unblockIp}
                onChange={(e) => setUnblockIp(e.target.value)}
                placeholder="192.168.1.100"
              />
              <button
                onClick={() => handleUnblock(null, unblockIp)}
                disabled={!unblockIp}
                className="btn-primary"
              >
                Unblock
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Database Maintenance */}
      <div className="metrics-card maintenance-card">
        <div className="maintenance-content">
          <div className="maintenance-icon">🧹</div>
          <div>
            <h3>Database Maintenance</h3>
            <p>Clean up old brute force records to optimize database performance and free up storage space. This will permanently delete records older than 30 days.</p>
            <button onClick={handleCleanup} className="btn-warning">
              🧹 Clean Up Records Older Than 30 Days
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BruteForcePanel;
