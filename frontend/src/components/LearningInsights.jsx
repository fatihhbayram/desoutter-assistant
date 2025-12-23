import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Dashboard.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://192.168.1.125:8000';

const LearningInsights = ({ token }) => {
    const [learningStats, setLearningStats] = useState(null);
    const [topSources, setTopSources] = useState([]);
    const [trainingStatus, setTrainingStatus] = useState(null);
    const [testQuery, setTestQuery] = useState('');
    const [recommendations, setRecommendations] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        loadAllData();
    }, []);

    const loadAllData = async () => {
        setLoading(true);
        try {
            await Promise.all([
                loadLearningStats(),
                loadTopSources(),
                loadTrainingStatus()
            ]);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const loadLearningStats = async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/admin/learning/stats`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            setLearningStats(response.data);
        } catch (err) {
            console.error('Error loading learning stats:', err);
        }
    };

    const loadTopSources = async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/admin/learning/top-sources`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            setTopSources(response.data.top_sources || []);
        } catch (err) {
            console.error('Error loading top sources:', err);
        }
    };

    const loadTrainingStatus = async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/admin/learning/training-status`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            setTrainingStatus(response.data);
        } catch (err) {
            console.error('Error loading training status:', err);
        }
    };

    const handleTestQuery = async () => {
        if (!testQuery.trim()) {
            alert('Please enter a test query');
            return;
        }

        try {
            // Extract keywords from query by splitting on spaces and filtering empty strings
            const keywords = testQuery.trim().split(/\s+/).filter(k => k.length > 0);

            const response = await axios.post(
                `${API_BASE_URL}/admin/learning/recommendations`,
                { keywords: keywords },
                { headers: { Authorization: `Bearer ${token}` } }
            );
            setRecommendations(response.data);
        } catch (err) {
            alert(`Error: ${err.response?.data?.detail || err.message}`);
        }
    };

    const handleScheduleRetraining = async () => {
        if (!window.confirm('Schedule embedding retraining? This may take several minutes.')) return;

        try {
            await axios.post(
                `${API_BASE_URL}/admin/learning/schedule-retraining`,
                {},
                { headers: { Authorization: `Bearer ${token}` } }
            );
            alert('Retraining scheduled successfully');
            loadTrainingStatus();
        } catch (err) {
            alert(`Error: ${err.message}`);
        }
    };

    const handleReset = async () => {
        if (!window.confirm('⚠️ WARNING: This will reset ALL learning data. Are you sure?')) return;
        if (!window.confirm('This action CANNOT be undone. Continue?')) return;

        try {
            await axios.post(
                `${API_BASE_URL}/admin/learning/reset`,
                {},
                { headers: { Authorization: `Bearer ${token}` } }
            );
            alert('Learning data reset successfully');
            loadAllData();
        } catch (err) {
            alert(`Error: ${err.message}`);
        }
    };

    if (loading) {
        return (
            <div className="loading">
                <div className="spinner"></div>
                <p>Loading learning insights...</p>
            </div>
        );
    }

    if (error) {
        return <div className="error">Error: {error}</div>;
    }

    return (
        <div className="learning-insights">
            <div className="dashboard-header">
                <h2>🧠 Learning Insights</h2>
                <div className="dashboard-actions">
                    <button onClick={loadAllData} className="btn-primary">🔄 Refresh</button>
                </div>
            </div>

            {/* Learning Stats Card */}
            <div className="metrics-card">
                <h3>📈 Learning Statistics</h3>
                {learningStats && (
                    <div className="stats-grid">
                        <div className="stat">
                            <span className="stat-label">Total Sources Learned</span>
                            <span className="stat-value">{learningStats.total_sources || 0}</span>
                        </div>
                        <div className="stat">
                            <span className="stat-label">High Confidence Sources</span>
                            <span className="stat-value">{learningStats.high_confidence_sources || 0}</span>
                        </div>
                        <div className="stat">
                            <span className="stat-label">Keyword Mappings</span>
                            <span className="stat-value">{learningStats.total_mappings || 0}</span>
                        </div>
                        <div className="stat">
                            <span className="stat-label">Success Rate</span>
                            <span className="stat-value">{learningStats.success_rate || 0}%</span>
                        </div>
                        <div className="stat">
                            <span className="stat-label">Events (Last Week)</span>
                            <span className="stat-value">{learningStats.events_last_week || 0}</span>
                        </div>
                    </div>
                )}
            </div>

            {/* Top Sources Table */}
            <div className="metrics-card">
                <h3>🏆 Top Performing Sources</h3>
                {topSources.length > 0 ? (
                    <table className="source-table">
                        <thead>
                            <tr>
                                <th>Source</th>
                                <th>Score</th>
                                <th>Positive</th>
                                <th>Negative</th>
                                <th>Confidence</th>
                            </tr>
                        </thead>
                        <tbody>
                            {topSources.slice(0, 10).map((source, idx) => (
                                <tr key={idx}>
                                    <td>{source.source_name || 'N/A'}</td>
                                    <td>{source.score ? source.score.toFixed(2) : '0.00'}</td>
                                    <td>{source.positive_signals || 0}</td>
                                    <td>{source.negative_signals || 0}</td>
                                    <td>
                                        <span className={`confidence-badge ${source.confidence || 'low'}`}>
                                            {source.confidence || 'low'}
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                ) : (
                    <p>No sources learned yet</p>
                )}
            </div>

            {/* Training Status Card */}
            <div className="metrics-card">
                <h3>🎓 Training Status</h3>
                {trainingStatus && (
                    <div className="training-status">
                        <div className="stats-grid">
                            <div className="stat">
                                <span className="stat-label">Total Training Samples</span>
                                <span className="stat-value">{trainingStatus.total_samples || 0}</span>
                            </div>
                            <div className="stat">
                                <span className="stat-label">Unused Samples</span>
                                <span className="stat-value">{trainingStatus.unused_samples || 0}</span>
                            </div>
                            <div className="stat">
                                <span className="stat-label">Samples (Last Week)</span>
                                <span className="stat-value">{trainingStatus.samples_last_week || 0}</span>
                            </div>
                            <div className="stat">
                                <span className="stat-label">Ready for Training</span>
                                <span className="stat-value">
                                    {trainingStatus.ready_for_training ? '✅ Yes' : '❌ No'}
                                </span>
                            </div>
                        </div>
                        <button
                            onClick={handleScheduleRetraining}
                            className="btn-primary"
                            disabled={!trainingStatus.ready_for_training}
                            style={{ marginTop: '15px' }}
                        >
                            Schedule Retraining
                        </button>
                    </div>
                )}
            </div>

            {/* Query Recommendations Tester */}
            <div className="metrics-card">
                <h3>🔮 Query Recommendations Tester</h3>
                <div className="enhancement-tester">
                    <div style={{ marginBottom: '15px' }}>
                        <input
                            type="text"
                            value={testQuery}
                            onChange={(e) => setTestQuery(e.target.value)}
                            placeholder="Enter a test query (e.g., 'motor grinding noise')"
                            style={{ width: '100%', padding: '10px', marginBottom: '10px' }}
                        />
                        <button onClick={handleTestQuery} className="btn-primary">
                            Get Recommendations
                        </button>
                    </div>

                    {recommendations && (
                        <div className="enhancement-result">
                            <h4>Recommendations:</h4>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                                <div>
                                    <strong>✅ Boost Sources:</strong>
                                    <ul>
                                        {recommendations.boost_sources && recommendations.boost_sources.length > 0 ? (
                                            recommendations.boost_sources.map((src, idx) => (
                                                <li key={idx}>{src}</li>
                                            ))
                                        ) : (
                                            <li>None</li>
                                        )}
                                    </ul>
                                </div>
                                <div>
                                    <strong>❌ Avoid Sources:</strong>
                                    <ul>
                                        {recommendations.avoid_sources && recommendations.avoid_sources.length > 0 ? (
                                            recommendations.avoid_sources.map((src, idx) => (
                                                <li key={idx}>{src}</li>
                                            ))
                                        ) : (
                                            <li>None</li>
                                        )}
                                    </ul>
                                </div>
                            </div>
                            <p style={{ marginTop: '10px' }}>
                                <strong>Confidence:</strong> {recommendations.confidence || 'N/A'}
                            </p>
                        </div>
                    )}
                </div>
            </div>

            {/* Danger Zone */}
            <div className="danger-zone">
                <h3>⚠️ Danger Zone</h3>
                <p>Reset all learning data. This action cannot be undone.</p>
                <button onClick={handleReset} className="btn-danger">
                    Reset Learning Data
                </button>
            </div>
        </div>
    );
};

export default LearningInsights;
