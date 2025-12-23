import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Dashboard.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://192.168.1.125:8000';

const DomainManagement = ({ token }) => {
    const [domainStats, setDomainStats] = useState(null);
    const [errorCodes, setErrorCodes] = useState([]);
    const [productSeries, setProductSeries] = useState([]);
    const [testQuery, setTestQuery] = useState('');
    const [enhancedQuery, setEnhancedQuery] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [activeSection, setActiveSection] = useState('stats');

    useEffect(() => {
        loadDomainStats();
    }, []);

    const loadDomainStats = async () => {
        setLoading(true);
        try {
            const response = await axios.get(`${API_BASE_URL}/admin/domain/stats`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            setDomainStats(response.data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const loadErrorCodes = async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/admin/domain/error-codes`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            setErrorCodes(response.data.error_codes || []);
            setActiveSection('error-codes');
        } catch (err) {
            alert(`Error: ${err.message}`);
        }
    };

    const loadProductSeries = async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/admin/domain/product-series`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            setProductSeries(response.data.product_series || []);
            setActiveSection('product-series');
        } catch (err) {
            alert(`Error: ${err.message}`);
        }
    };

    const handleEnhanceQuery = async () => {
        if (!testQuery.trim()) {
            alert('Please enter a test query');
            return;
        }

        try {
            const response = await axios.post(
                `${API_BASE_URL}/admin/domain/enhance-query`,
                { query: testQuery },
                { headers: { Authorization: `Bearer ${token}` } }
            );
            setEnhancedQuery(response.data);
        } catch (err) {
            alert(`Error: ${err.message}`);
        }
    };

    if (loading) {
        return (
            <div className="loading">
                <div className="spinner"></div>
                <p>Loading domain management...</p>
            </div>
        );
    }

    if (error) {
        return <div className="error">Error: {error}</div>;
    }

    return (
        <div className="domain-management">
            <div className="dashboard-header">
                <h2>📚 Domain Management</h2>
                <div className="dashboard-actions">
                    <button onClick={loadDomainStats} className="btn-primary">🔄 Refresh</button>
                </div>
            </div>

            {/* Domain Stats Card */}
            <div className="metrics-card">
                <h3>📊 Domain Vocabulary Statistics</h3>
                {domainStats && (
                    <div className="stats-grid">
                        <div className="stat">
                            <span className="stat-label">Total Synonyms</span>
                            <span className="stat-value">{domainStats.total_synonyms || 0}</span>
                        </div>
                        <div className="stat">
                            <span className="stat-label">Error Codes</span>
                            <span className="stat-value">{domainStats.total_error_codes || 0}</span>
                        </div>
                        <div className="stat">
                            <span className="stat-label">Product Series</span>
                            <span className="stat-value">{domainStats.total_product_series || 0}</span>
                        </div>
                        <div className="stat">
                            <span className="stat-label">Components</span>
                            <span className="stat-value">{domainStats.total_components || 0}</span>
                        </div>
                        <div className="stat">
                            <span className="stat-label">Specifications</span>
                            <span className="stat-value">{domainStats.total_specifications || 0}</span>
                        </div>
                    </div>
                )}
            </div>

            {/* Query Enhancement Tester */}
            <div className="metrics-card">
                <h3>🔮 Query Enhancement Tester</h3>
                <div className="enhancement-tester">
                    <div style={{ marginBottom: '15px' }}>
                        <input
                            type="text"
                            value={testQuery}
                            onChange={(e) => setTestQuery(e.target.value)}
                            placeholder="Enter a test query (e.g., 'motor grinding noise E47')"
                            style={{ width: '100%', padding: '10px', marginBottom: '10px' }}
                        />
                        <button onClick={handleEnhanceQuery} className="btn-primary">
                            Enhance Query
                        </button>
                    </div>

                    {enhancedQuery && (
                        <div className="enhancement-result">
                            <div className="query-diff">
                                <div>
                                    <h4>Original Query:</h4>
                                    <p className="original">{enhancedQuery.original || testQuery}</p>
                                </div>
                                <div>
                                    <h4>Enhanced Query:</h4>
                                    <p className="enhanced">{enhancedQuery.enhanced || 'N/A'}</p>
                                </div>
                            </div>

                            {enhancedQuery.entities && Object.keys(enhancedQuery.entities).length > 0 && (
                                <div style={{ marginTop: '15px' }}>
                                    <h4>Extracted Entities:</h4>
                                    <ul>
                                        {Object.entries(enhancedQuery.entities).map(([key, values]) => (
                                            <li key={key}>
                                                <strong>{key}:</strong> {Array.isArray(values) ? values.join(', ') : values}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            {enhancedQuery.added_synonyms && enhancedQuery.added_synonyms.length > 0 && (
                                <div style={{ marginTop: '15px' }}>
                                    <h4>Added Synonyms:</h4>
                                    <p>{enhancedQuery.added_synonyms.join(', ')}</p>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>

            {/* Vocabulary Browser */}
            <div className="metrics-card">
                <h3>📖 Vocabulary Browser</h3>

                <div className="vocabulary-tabs">
                    <button
                        className={activeSection === 'stats' ? 'active' : ''}
                        onClick={() => setActiveSection('stats')}
                    >
                        Stats
                    </button>
                    <button
                        className={activeSection === 'error-codes' ? 'active' : ''}
                        onClick={loadErrorCodes}
                    >
                        Error Codes
                    </button>
                    <button
                        className={activeSection === 'product-series' ? 'active' : ''}
                        onClick={loadProductSeries}
                    >
                        Product Series
                    </button>
                </div>

                <div className="vocabulary-content">
                    {activeSection === 'stats' && (
                        <p>Select a tab above to view vocabulary data</p>
                    )}

                    {activeSection === 'error-codes' && (
                        <div>
                            {errorCodes.length > 0 ? (
                                <table className="query-table">
                                    <thead>
                                        <tr>
                                            <th>Code</th>
                                            <th>Description</th>
                                            <th>Category</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {errorCodes.map((code, idx) => (
                                            <tr key={idx}>
                                                <td><strong>{code.code || 'N/A'}</strong></td>
                                                <td>{code.description || 'N/A'}</td>
                                                <td>{code.category || 'N/A'}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            ) : (
                                <p>No error codes available</p>
                            )}
                        </div>
                    )}

                    {activeSection === 'product-series' && (
                        <div>
                            {productSeries.length > 0 ? (
                                <table className="query-table">
                                    <thead>
                                        <tr>
                                            <th>Series Name</th>
                                            <th>Tool Type</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {productSeries.map((series, idx) => (
                                            <tr key={idx}>
                                                <td><strong>{series.series_name || 'N/A'}</strong></td>
                                                <td>{series.tool_type || 'N/A'}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            ) : (
                                <p>No product series available</p>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default DomainManagement;
