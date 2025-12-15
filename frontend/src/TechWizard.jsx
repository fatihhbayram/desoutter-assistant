import React, { useState, useMemo } from 'react';
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://192.168.1.125:8000';

/**
 * Tech Wizard Component
 * Step-by-step wizard for technicians to diagnose product problems
 * Step 1: Search & Filter Products
 * Step 2: Select Product
 * Step 3: Describe Fault & Get Diagnosis
 * Step 4: View Results & Feedback
 */
export const TechWizard = ({ token }) => {
  // Wizard State
  const [step, setStep] = useState(1);
  const [products, setProducts] = useState([]);
  const [productsLoaded, setProductsLoaded] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [faultDescription, setFaultDescription] = useState('');
  const [language, setLanguage] = useState('en');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  // Search & Filter
  const [search, setSearch] = useState('');
  const [filterSeries, setFilterSeries] = useState('');
  const [filterWireless, setFilterWireless] = useState(false);
  const [filterType, setFilterType] = useState('all'); // 'all', 'tool', 'cvi3_controller'

  // Load products once
  React.useEffect(() => {
    if (!productsLoaded && step >= 1) {
      loadProducts();
    }
  }, [productsLoaded, step]);

  const loadProducts = async () => {
    try {
      const response = await axios.get(`${API_BASE}/products`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      setProducts(response.data.products || []);
      setProductsLoaded(true);
    } catch (error) {
      console.error('Error loading products:', error);
    }
  };

  // Filter products
  const filteredProducts = useMemo(() => {
    return products.filter(p => {
      const matchesSearch = !search || 
        p.model_name?.toLowerCase().includes(search.toLowerCase()) ||
        p.part_number?.toLowerCase().includes(search.toLowerCase());
      
      const matchesSeries = !filterSeries || p.series_name === filterSeries;
      
      const matchesWireless = !filterWireless || 
        p.wireless_communication?.toLowerCase().includes('yes');
      
      const matchesType = filterType === 'all' || (p.product_type || 'tool') === filterType;
      
      return matchesSearch && matchesSeries && matchesWireless && matchesType;
    });
  }, [products, search, filterSeries, filterWireless, filterType]);

  // Get unique series for filter (dynamic based on product type)
  const uniqueSeries = useMemo(() => {
    let filtered = products;
    
    // Filter by type
    if (filterType === 'tool') {
      filtered = products.filter(p => !p.product_type || p.product_type === 'tool');
    } else if (filterType === 'cvi3_controller') {
      filtered = products.filter(p => p.product_type === 'cvi3_controller');
    }
    
    // Get unique series/category field
    const series = filtered
      .map(p => p.series_name || p.category)
      .filter(Boolean);
    
    return [...new Set(series)].sort();
  }, [products, filterType]);

  // Handle diagnosis
  const handleDiagnose = async () => {
    if (!selectedProduct || !faultDescription) return;

    setLoading(true);
    try {
      const response = await axios.post(
        `${API_BASE}/diagnose`,
        {
          product_id: selectedProduct.product_id,
          part_number: selectedProduct.part_number,
          model_name: selectedProduct.model_name,
          fault_description: faultDescription,
          language
        },
        {
          headers: { 'Authorization': `Bearer ${token}` }
        }
      );
      
      setResult(response.data);
      setFeedbackSubmitted(false);
      setStep(4);
    } catch (error) {
      console.error('Error getting diagnosis:', error);
      alert('Failed to get diagnosis. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Handle feedback
  const handleFeedback = async (feedbackType) => {
    if (!result?.diagnosis_id) return;

    try {
      await axios.post(
        `${API_BASE}/diagnose/feedback`,
        {
          diagnosis_id: result.diagnosis_id,
          feedback_type: feedbackType,
          negative_reason: null,
          user_comment: null,
          correct_solution: null
        },
        {
          headers: { 'Authorization': `Bearer ${token}` }
        }
      );
      setFeedbackSubmitted(true);
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
  };

  // Progress bar
  const progressSteps = ['Search', 'Select', 'Diagnose', 'Results'];
  
  return (
    <div className="wizard-container">
      {/* Progress Bar */}
      <div className="wizard-progress">
        <div className="progress-bar">
          <div 
            className="progress-fill" 
            style={{ width: `${(step / 4) * 100}%` }}
          ></div>
        </div>
        <div className="progress-steps">
          {progressSteps.map((s, i) => (
            <div
              key={i}
              className={`progress-step ${step > i + 1 ? 'completed' : ''} ${step === i + 1 ? 'active' : ''}`}
              onClick={() => {
                // Allow going back to previous steps
                if (i + 1 < step) {
                  setStep(i + 1);
                }
              }}
            >
              <span className="step-number">{i + 1}</span>
              <span className="step-label">{s}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Step Content */}
      <div className="wizard-content">
        {/* Step 1: Search & Filter */}
        {step === 1 && (
          <div className="wizard-step step-1">
            <div className="step-header">
              <h2>🔍 Search for Products</h2>
              <p>Find the tool you need to diagnose</p>
            </div>

            <div className="step-body">
              {/* Search Box */}
              <div className="search-section">
                <input
                  type="search"
                  placeholder="Search by model name or part number..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="search-input"
                  autoFocus
                />
                <span className="search-icon">🔍</span>
              </div>

              {/* Filters */}
              <div className="filters-section">
                {/* Product Series/Category Filter */}
                <div className="filter-group">
                  <label>Product {filterType === 'cvi3_controller' ? 'Category' : 'Series'}</label>
                  <select 
                    value={filterSeries}
                    onChange={(e) => setFilterSeries(e.target.value)}
                  >
                    <option value="">All {filterType === 'cvi3_controller' ? 'Categories' : 'Series'}</option>
                    {uniqueSeries.map(s => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </div>

                <div className="filter-group">
                  <label className="checkbox">
                    <input
                      type="checkbox"
                      checked={filterWireless}
                      onChange={(e) => setFilterWireless(e.target.checked)}
                    />
                    <span>📶 Wireless Only</span>
                  </label>
                </div>

                <div className="filter-group">
                  <label>Product Type</label>
                  <div className="type-buttons">
                    <button 
                      className={`type-btn ${filterType === 'all' ? 'active' : ''}`}
                      onClick={() => setFilterType('all')}
                    >
                      All ({products.length})
                    </button>
                    <button 
                      className={`type-btn ${filterType === 'tool' ? 'active' : ''}`}
                      onClick={() => setFilterType('tool')}
                    >
                      Tools ({products.filter(p => !p.product_type || p.product_type === 'tool').length})
                    </button>
                    <button 
                      className={`type-btn ${filterType === 'cvi3_controller' ? 'active' : ''}`}
                      onClick={() => setFilterType('cvi3_controller')}
                    >
                      CVI3 Units ({products.filter(p => p.product_type === 'cvi3_controller').length})
                    </button>
                  </div>
                </div>
              </div>

              {/* Results Count */}
              <div className="results-count">
                <span className="count-badge">{filteredProducts.length}</span>
                <span className="count-text">products found</span>
              </div>

              {/* Quick Preview */}
              {filteredProducts.length > 0 && (
                <div className="quick-preview">
                  <p className="preview-hint">Click Next to view and select a product</p>
                </div>
              )}

              {filteredProducts.length === 0 && (
                <div className="no-results">
                  <p>❌ No products found. Try different search terms.</p>
                </div>
              )}
            </div>

            {/* Step Actions */}
            <div className="step-actions">
              <button 
                className="btn-primary"
                onClick={() => setStep(2)}
                disabled={filteredProducts.length === 0}
              >
                Next: Select Product →
              </button>
            </div>
          </div>
        )}

        {/* Step 2: Select Product */}
        {step === 2 && (
          <div className="wizard-step step-2">
            <div className="step-header">
              <h2>📦 Select a Product</h2>
              <p>Choose the tool you want to diagnose ({filteredProducts.length} available)</p>
            </div>

            <div className="step-body">
              {/* Back to Search */}
              <button 
                className="back-to-search"
                onClick={() => setStep(1)}
              >
                ← Change Search
              </button>

              {/* Product Grid */}
              <div className="products-grid">
                {filteredProducts.map(product => (
                  <div
                    key={product.product_id}
                    className={`product-item ${selectedProduct?.product_id === product.product_id ? 'selected' : ''}`}
                    onClick={() => setSelectedProduct(product)}
                  >
                    {/* Product Image */}
                    <div className="product-img">
                      {product.image_url ? (
                        <img src={product.image_url} alt={product.model_name} />
                      ) : (
                        <div className="no-image">📷</div>
                      )}
                    </div>

                    {/* Product Info */}
                    <div className="product-info">
                      <div className="product-header">
                        <h3 className="product-name">{product.model_name}</h3>
                        <span className={`product-type-badge ${product.product_type === 'cvi3_controller' ? 'cvi3' : 'tool'}`}>
                          {product.product_type === 'cvi3_controller' ? '⚙️ CVI3' : '🔧 Tool'}
                        </span>
                      </div>
                      <div className="product-meta">
                        <span className="part-number">{product.part_number}</span>
                        {product.series_name && (
                          <span className="series-tag">{product.series_name}</span>
                        )}
                      </div>

                      {/* Quick specs */}
                      <div className="product-specs">
                        {product.wireless_communication?.toLowerCase().includes('yes') && (
                          <span className="spec-badge">📶 Wireless</span>
                        )}
                        {product.min_torque && product.max_torque && (
                          <span className="spec-badge">⚙️ {product.min_torque}-{product.max_torque}</span>
                        )}
                      </div>
                    </div>

                    {/* Selection Indicator */}
                    {selectedProduct?.product_id === product.product_id && (
                      <div className="selection-indicator">✅ Selected</div>
                    )}
                  </div>
                ))}
              </div>

              {filteredProducts.length === 0 && (
                <div className="no-results">
                  <p>No products to display. Go back and adjust your search.</p>
                </div>
              )}
            </div>

            {/* Step Actions */}
            <div className="step-actions">
              <button 
                className="btn-secondary"
                onClick={() => setStep(1)}
              >
                ← Back
              </button>
              <button 
                className="btn-primary"
                onClick={() => setStep(3)}
                disabled={!selectedProduct}
              >
                Next: Describe Problem →
              </button>
            </div>
          </div>
        )}

        {/* Step 3: Diagnose */}
        {step === 3 && (
          <div className="wizard-step step-3">
            <div className="step-header">
              <h2>🩺 Describe the Problem</h2>
              <p>Tell us what's wrong with the {selectedProduct?.model_name}</p>
            </div>

            <div className="step-body">
              {/* Selected Product Summary */}
              <div className="product-summary">
                <div className="summary-img">
                  {selectedProduct?.image_url ? (
                    <img src={selectedProduct.image_url} alt={selectedProduct?.model_name} />
                  ) : (
                    <div className="no-image-sm">📷</div>
                  )}
                </div>
                <div className="summary-info">
                  <h3>{selectedProduct?.model_name}</h3>
                  <p>{selectedProduct?.part_number}</p>
                  <button 
                    className="change-product"
                    onClick={() => setStep(2)}
                  >
                    Change Product
                  </button>
                </div>
              </div>

              {/* Fault Description */}
              <div className="form-group">
                <label>Describe the fault or problem *</label>
                <textarea
                  value={faultDescription}
                  onChange={(e) => setFaultDescription(e.target.value)}
                  placeholder="Example: Motor makes grinding noise when turned on, battery drains quickly, tool won't start..."
                  rows="5"
                  disabled={loading}
                  className="fault-textarea"
                />
                <span className="char-count">{faultDescription.length} characters</span>
              </div>

              {/* Language Selection */}
              <div className="form-group">
                <label>Response Language</label>
                <div className="language-selector">
                  <button 
                    className={`lang-btn ${language === 'en' ? 'active' : ''}`}
                    onClick={() => setLanguage('en')}
                    disabled={loading}
                  >
                    🇬🇧 English
                  </button>
                  <button 
                    className={`lang-btn ${language === 'tr' ? 'active' : ''}`}
                    onClick={() => setLanguage('tr')}
                    disabled={loading}
                  >
                    🇹🇷 Türkçe
                  </button>
                </div>
              </div>
            </div>

            {/* Step Actions */}
            <div className="step-actions">
              <button 
                className="btn-secondary"
                onClick={() => setStep(2)}
                disabled={loading}
              >
                ← Back
              </button>
              <button 
                className="btn-primary btn-diagnose"
                onClick={handleDiagnose}
                disabled={!faultDescription.trim() || loading}
              >
                {loading ? (
                  <>⏳ Analyzing...</>
                ) : (
                  <>Get Repair Suggestion →</>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Step 4: Results */}
        {step === 4 && result && (
          <div className="wizard-step step-4">
            <div className="step-header">
              <h2>✅ Repair Suggestion</h2>
              <div className="confidence-badge" style={{
                backgroundColor: result.confidence === 'high' ? '#10b981' : 
                                result.confidence === 'medium' ? '#f59e0b' : '#ef4444'
              }}>
                {result.confidence === 'high' && '🟢'}
                {result.confidence === 'medium' && '🟡'}
                {result.confidence === 'low' && '🔴'}
                <span>{result.confidence} confidence</span>
              </div>
            </div>

            <div className="step-body">
              {/* Main Suggestion */}
              <div className="suggestion-box">
                <div className="suggestion-text">
                  {result.suggestion}
                </div>
                <div className="suggestion-meta">
                  ⚡ Response time: {(result.response_time_ms / 1000).toFixed(1)}s
                </div>
              </div>

              {/* Sources */}
              {result.sources && result.sources.length > 0 && (
                <div className="sources-section">
                  <h4>📚 Related Documents ({result.sources.length})</h4>
                  <div className="sources-list">
                    {result.sources.slice(0, 3).map((source, idx) => (
                      <div key={idx} className="source-item">
                        <div className="source-details">
                          <span className="source-name">📄 {source.source}</span>
                          <span className="source-similarity">Similarity: {source.similarity}</span>
                        </div>
                        <button 
                          className="open-doc-btn"
                          onClick={() => window.open(`${API_BASE}/documents/download/${encodeURIComponent(source.source)}`, '_blank')}
                        >
                          Open
                        </button>
                      </div>
                    ))}
                  </div>
                  {result.sources.length > 3 && (
                    <details className="more-sources">
                      <summary>+{result.sources.length - 3} more documents</summary>
                      <div className="sources-list">
                        {result.sources.slice(3).map((source, idx) => (
                          <div key={idx} className="source-item">
                            <span>{source.source}</span>
                            <button 
                              onClick={() => window.open(`${API_BASE}/documents/download/${encodeURIComponent(source.source)}`, '_blank')}
                            >
                              Open
                            </button>
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                </div>
              )}

              {/* Feedback */}
              <div className="feedback-section">
                <h4>Was this suggestion helpful?</h4>
                {!feedbackSubmitted ? (
                  <div className="feedback-buttons">
                    <button 
                      className="feedback-btn positive"
                      onClick={() => handleFeedback('positive')}
                    >
                      👍 Yes, Helpful
                    </button>
                    <button 
                      className="feedback-btn negative"
                      onClick={() => handleFeedback('negative')}
                    >
                      👎 No, Try Different
                    </button>
                  </div>
                ) : (
                  <div className="feedback-success">
                    ✅ Thank you! Your feedback helps us improve.
                  </div>
                )}
              </div>
            </div>

            {/* Step Actions */}
            <div className="step-actions">
              <button 
                className="btn-secondary"
                onClick={() => {
                  setStep(1);
                  setSelectedProduct(null);
                  setFaultDescription('');
                  setResult(null);
                  setSearch('');
                  setFilterSeries('');
                  setFilterWireless(false);
                }}
              >
                ← Start New Diagnosis
              </button>
              <button 
                className="btn-primary"
                onClick={() => setStep(3)}
              >
                Diagnose Another Problem →
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TechWizard;
