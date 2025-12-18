/**
 * =============================================================================
 * Desoutter Repair Assistant - Main React Application
 * =============================================================================
 * 
 * This is the main application component providing:
 * - Authentication: Login/logout with JWT tokens
 * - Role-based UI: Admin panel vs Technician interface
 * - Product browsing: Search, filter, and select products
 * - AI Diagnosis: Get repair suggestions from RAG-powered backend
 * - Document Management: Upload PDF, Word, PowerPoint files for RAG knowledge base (admin only)
 * - User Management: CRUD operations for users (admin only)
 * 
 * Architecture:
 * - Uses axios for API communication with JWT auth interceptor
 * - State managed with React useState hooks
 * - Responsive design with CSS grid/flexbox
 * 
 * API Endpoints Used:
 * - POST /auth/login - User authentication
 * - GET /products - List products
 * - GET /stats - System statistics
 * - POST /diagnose - AI repair suggestion
 * - GET/POST/DELETE /admin/users - User management
 * - GET/POST/DELETE /admin/documents - Document management
 * - POST /admin/documents/ingest - RAG ingestion
 * 
 * =============================================================================
 */

import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './App.css';
import TechWizard from './TechWizard';
import './TechWizard.css';

// =============================================================================
// API CONFIGURATION
// =============================================================================

// API base URL - uses environment variable or defaults to Proxmox host
const API_BASE = import.meta.env.VITE_API_URL || 'http://192.168.1.125:8000';

// Create axios instance with base URL
const api = axios.create({ baseURL: API_BASE });

// Axios request interceptor - automatically adds JWT token to all requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Axios response interceptor - handle token expiry and auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle 401 Unauthorized - token expired or invalid
    if (error.response?.status === 401) {
      // Clear stored auth data
      localStorage.removeItem('auth_token');
      localStorage.removeItem('auth_role');
      localStorage.removeItem('auth_username');
      // Reload page to show login
      window.location.reload();
    }
    return Promise.reject(error);
  }
);

// =============================================================================
// MAIN APPLICATION COMPONENT
// =============================================================================

function App() {
  // ---------------------------------------------------------------------------
  // STATE MANAGEMENT
  // ---------------------------------------------------------------------------
  
  // Products data from API
  const [products, setProducts] = useState([]);
  
  // Authentication state
  const [auth, setAuth] = useState({ loggedIn: false, role: null, username: '' });
  const [loginForm, setLoginForm] = useState({ username: '', password: '' });
  
  // App initialization state - true while checking stored auth on mount
  const [initializing, setInitializing] = useState(true);
  
  // Product selection and diagnosis
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [search, setSearch] = useState('');
  const [faultDescription, setFaultDescription] = useState('');
  const [language, setLanguage] = useState('en');  // Response language: 'en' or 'tr'
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);  // AI diagnosis result
  const [stats, setStats] = useState(null);    // System statistics
  
  // Feedback state
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);
  const [feedbackReason, setFeedbackReason] = useState('');
  const [feedbackComment, setFeedbackComment] = useState('');
  const [retryLoading, setRetryLoading] = useState(false);
  const [sourceRelevance, setSourceRelevance] = useState({});  // { source_name: true/false }
  
  // Toast notifications
  const [toast, setToast] = useState(null);
  
  // Product filtering and pagination
  const [filterSeries, setFilterSeries] = useState('');
  const [filterWirelessOnly, setFilterWirelessOnly] = useState(false);
  const [torqueMin, setTorqueMin] = useState('');
  const [torqueMax, setTorqueMax] = useState('');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(12);
  const [viewMode, setViewMode] = useState('grid');  // 'grid' or 'list'
  
  // Admin panel states - User Management
  const [adminUsers, setAdminUsers] = useState([]);
  const [adminNewUser, setAdminNewUser] = useState({ username: '', password: '', role: 'technician' });
  const [adminLoadingUsers, setAdminLoadingUsers] = useState(false);
  const [adminScraping, setAdminScraping] = useState(false);
  
  // Admin panel states - Document Management (RAG)
  const [adminDocuments, setAdminDocuments] = useState([]);
  const [adminLoadingDocs, setAdminLoadingDocs] = useState(false);
  const [adminUploading, setAdminUploading] = useState(false);
  const [adminIngesting, setAdminIngesting] = useState(false);
  const [adminUploadType, setAdminUploadType] = useState('manual');  // 'manual' or 'bulletin'

  // Admin panel states - Dashboard
  const [dashboardData, setDashboardData] = useState(null);
  const [dashboardLoading, setDashboardLoading] = useState(false);
  const [adminTab, setAdminTab] = useState('dashboard'); // 'dashboard', 'users', 'documents'

  // ---------------------------------------------------------------------------
  // SIDE EFFECTS - Data Loading
  // ---------------------------------------------------------------------------
  
  // Check authentication on app mount - restore session from localStorage
  useEffect(() => {
    const checkAuthOnMount = async () => {
      const token = localStorage.getItem('auth_token');
      const role = localStorage.getItem('auth_role');
      const username = localStorage.getItem('auth_username');
      
      if (token && role && username) {
        try {
          // Verify token is still valid with backend
          await api.get('/auth/me');
          // Token is valid, restore auth state
          setAuth({ loggedIn: true, role, username });
        } catch (err) {
          // Token invalid/expired - clear storage
          console.warn('Stored token invalid, clearing auth');
          localStorage.removeItem('auth_token');
          localStorage.removeItem('auth_role');
          localStorage.removeItem('auth_username');
        }
      }
      setInitializing(false);
    };
    
    checkAuthOnMount();
  }, []);
  
  // Load data after successful login
  useEffect(() => {
    // Only load data after login
    if (auth.loggedIn) {
      loadProducts();  // Load product catalog
      loadStats();     // Load system statistics
      
      // Load admin-specific data if user is admin
      if (auth.role === 'admin') {
        loadAdminUsers();      // Load user list
        loadAdminDocuments();  // Load RAG documents
        loadDashboard();       // Load dashboard stats
      }
    }
  }, [auth.loggedIn, auth.role]);

  // ---------------------------------------------------------------------------
  // COMPUTED VALUES - Memoized Filters and Facets
  // ---------------------------------------------------------------------------
  
  // Compute filter facets (series, torque range) from products
  // This enables dynamic filter options based on available products
  const facets = React.useMemo(() => {
    const series = new Set();
    let tmin = Infinity;
    let tmax = -Infinity;
    for (const p of products) {
      if (p.series_name) series.add(p.series_name);
      const mn = parseFloat(p.min_torque);
      const mx = parseFloat(p.max_torque);
      if (!Number.isNaN(mn)) tmin = Math.min(tmin, mn);
      if (!Number.isNaN(mx)) tmax = Math.max(tmax, mx);
    }
    return {
      series: Array.from(series).sort(),
      tmin: tmin === Infinity ? 0 : tmin,
      tmax: tmax === -Infinity ? 0 : tmax,
    };
  }, [products]);

  // Initialize torque range sliders when products/facets load
  useEffect(() => {
    if (facets.tmax > 0 && torqueMin === '' && torqueMax === '') {
      setTorqueMin(String(Math.floor(facets.tmin)));
      setTorqueMax(String(Math.ceil(facets.tmax)));
    }
  }, [facets]);

  // Filter products based on search query and selected filters
  // Returns filtered array of products matching all criteria

  const filteredProducts = React.useMemo(() => {
    let out = products;
    const q = (search || '').toLowerCase();
    if (q) {
      out = out.filter(p => (
        (p.model_name || '').toLowerCase().includes(q) ||
        (p.part_number || '').toLowerCase().includes(q)
      ));
    }
    if (filterSeries) {
      out = out.filter(p => (p.series_name || '') === filterSeries);
    }
    if (filterWirelessOnly) {
      out = out.filter(p => String(p.wireless_communication || '').toLowerCase().includes('yes') || String(p.wireless_communication || '').toLowerCase().includes('true'));
    }
    const tmin = parseFloat(torqueMin);
    const tmax = parseFloat(torqueMax);
    if (!Number.isNaN(tmin) || !Number.isNaN(tmax)) {
      out = out.filter(p => {
        const mn = parseFloat(p.min_torque);
        const mx = parseFloat(p.max_torque);
        const has = !(Number.isNaN(mn) && Number.isNaN(mx));
        if (!has) return true; // if no torque info, keep it
        const pmn = Number.isNaN(mn) ? mx : mn;
        const pmx = Number.isNaN(mx) ? mn : mx;
        const lo = Number.isNaN(tmin) ? -Infinity : tmin;
        const hi = Number.isNaN(tmax) ? Infinity : tmax;
        // overlap of [pmn, pmx] and [lo, hi]
        return pmx >= lo && pmn <= hi;
      });
    }
    return out;
  }, [products, search, filterSeries, filterWirelessOnly, torqueMin, torqueMax]);

  // Pagination calculations
  const totalPages = Math.max(1, Math.ceil(filteredProducts.length / pageSize));
  
  // Get current page of products (sliced from filtered results)
  const pagedProducts = React.useMemo(() => {
    const p = Math.min(page, totalPages);
    const start = (p - 1) * pageSize;
    return filteredProducts.slice(start, start + pageSize);
  }, [filteredProducts, page, pageSize, totalPages]);

  // ---------------------------------------------------------------------------
  // DATA LOADING FUNCTIONS
  // ---------------------------------------------------------------------------

  /**
   * Load all products from the API
   * Populates the products state for display and filtering
   */
  const loadProducts = async () => {
    try {
      const response = await api.get('/products');
      setProducts(response.data.products);
    } catch (error) {
      console.error('Error loading products:', error);
      setToast({ type: 'error', message: 'Failed to load products' });
    }
  };

  /**
   * Load system statistics (product count, document count, RAG status)
   */
  const loadStats = async () => {
    try {
      const response = await api.get('/stats');
      setStats(response.data);
    } catch (error) {
      console.error('Error loading stats:', error);
      setToast({ type: 'error', message: 'Failed to load stats' });
    }
  };

  // ---------------------------------------------------------------------------
  // ADMIN FUNCTIONS - User Management
  // ---------------------------------------------------------------------------

  /**
   * Load list of users (admin only)
   */
  const loadAdminUsers = async () => {
    setAdminLoadingUsers(true);
    try {
      const res = await api.get('/admin/users');
      setAdminUsers(res.data.users || []);
    } catch (err) {
      console.log('Users endpoint not available yet');
      setAdminUsers([]);
    } finally {
      setAdminLoadingUsers(false);
    }
  };

  /**
   * Load dashboard statistics (admin only)
   */
  const loadDashboard = async () => {
    setDashboardLoading(true);
    try {
      const res = await api.get('/admin/dashboard');
      setDashboardData(res.data);
    } catch (err) {
      console.error('Dashboard load error:', err);
      setDashboardData(null);
    } finally {
      setDashboardLoading(false);
    }
  };

  /**
   * Handle form submission to create a new user
   */
  const handleAdminAddUser = async (e) => {
    e.preventDefault();
    if (!adminNewUser.username || !adminNewUser.password) {
      setToast({ type: 'error', message: 'Username and password required' });
      return;
    }
    try {
      await api.post('/admin/users', adminNewUser);
      setToast({ type: 'info', message: `User ${adminNewUser.username} created` });
      setAdminNewUser({ username: '', password: '', role: 'technician' });
      loadAdminUsers();  // Refresh user list
    } catch (err) {
      setToast({ type: 'error', message: err.response?.data?.detail || 'Failed to create user' });
    }
  };

  /**
   * Handle user deletion with confirmation
   */
  const handleAdminDeleteUser = async (username) => {
    if (!confirm(`Delete user "${username}"?`)) return;
    try {
      await api.delete(`/admin/users/${username}`);
      setToast({ type: 'info', message: `User ${username} deleted` });
      loadAdminUsers();  // Refresh user list
    } catch (err) {
      setToast({ type: 'error', message: err.response?.data?.detail || 'Failed to delete user' });
    }
  };

  /**
   * Trigger web scraping to refresh product data
   */
  const handleAdminScrape = async () => {
    setAdminScraping(true);
    try {
      const res = await api.post('/admin/scrape');
      setToast({ type: 'info', message: res.data?.message || 'Scraping started' });
      // Refresh data after 5 seconds (scraping is async)
      setTimeout(() => {
        loadProducts();
        loadStats();
      }, 5000);
    } catch (err) {
      setToast({ type: 'error', message: err.response?.data?.detail || 'Scrape failed' });
    } finally {
      setAdminScraping(false);
    }
  };

  // ---------------------------------------------------------------------------
  // ADMIN FUNCTIONS - Document Management (RAG)
  // ---------------------------------------------------------------------------

  /**
   * Load list of uploaded documents for RAG
   */
  const loadAdminDocuments = async () => {
    setAdminLoadingDocs(true);
    try {
      const res = await api.get('/admin/documents');
      setAdminDocuments(res.data.documents || []);
    } catch (err) {
      console.log('Documents endpoint not available yet');
      setAdminDocuments([]);
    } finally {
      setAdminLoadingDocs(false);
    }
  };

  /**
   * Handle document file upload for RAG knowledge base
   * Validates file type (PDF, DOCX, PPTX) and sends to backend
   */
  const handleAdminUploadDocument = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    // Validate file type - PDF, Word, PowerPoint allowed
    const allowedExtensions = ['.pdf', '.docx', '.doc', '.pptx', '.ppt'];
    const fileExt = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
    if (!allowedExtensions.includes(fileExt)) {
      setToast({ type: 'error', message: 'Supported formats: PDF, Word (DOCX), PowerPoint (PPTX)' });
      return;
    }
    
    setAdminUploading(true);
    try {
      // Prepare multipart form data
      const formData = new FormData();
      formData.append('file', file);
      formData.append('doc_type', adminUploadType);
      
      await api.post('/admin/documents/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      setToast({ type: 'info', message: `${file.name} uploaded successfully` });
      loadAdminDocuments();  // Refresh document list
      e.target.value = '';   // Reset file input for next upload
    } catch (err) {
      setToast({ type: 'error', message: err.response?.data?.detail || 'Upload failed' });
    } finally {
      setAdminUploading(false);
    }
  };

  /**
   * Handle document deletion with confirmation
   * Also removes document from vector database
   */
  const handleAdminDeleteDocument = async (doc) => {
    if (!confirm(`Delete document "${doc.filename}"?`)) return;
    try {
      await api.delete(`/admin/documents/${doc.type}/${doc.filename}`);
      setToast({ type: 'info', message: `${doc.filename} deleted` });
      loadAdminDocuments();  // Refresh document list
    } catch (err) {
      setToast({ type: 'error', message: err.response?.data?.detail || 'Delete failed' });
    }
  };

  /**
   * Trigger document ingestion to update RAG vector database
   * Processes all PDFs: extracts text, chunks, generates embeddings
   */
  const handleAdminIngestDocuments = async () => {
    if (!confirm('This will re-process all documents and update the RAG database. Continue?')) return;
    setAdminIngesting(true);
    try {
      const res = await api.post('/admin/documents/ingest');
      setToast({ type: 'info', message: res.data?.message || 'Ingestion complete' });
      loadStats();  // Refresh stats to show new document count
    } catch (err) {
      setToast({ type: 'error', message: err.response?.data?.detail || 'Ingestion failed' });
    } finally {
      setAdminIngesting(false);
    }
  };

  // ---------------------------------------------------------------------------
  // AUTHENTICATION FUNCTIONS
  // ---------------------------------------------------------------------------

  /**
   * Handle login form submission
   * Stores JWT token and user info in localStorage on success
   */
  const handleLogin = async (e) => {
    e?.preventDefault();
    const { username, password } = loginForm;
    if (!username || !password) {
      setToast({ type: 'error', message: 'Please enter username and password' });
      return;
    }
    try {
      const res = await api.post('/auth/login', { username, password });
      const { access_token, role } = res.data || {};
      if (!access_token) throw new Error('No token received');
      
      // Persist auth data to localStorage for session persistence
      localStorage.setItem('auth_token', access_token);
      localStorage.setItem('auth_role', role || 'technician');
      localStorage.setItem('auth_username', username);
      
      setAuth({ loggedIn: true, role: role || 'technician', username });
      setToast({ type: 'info', message: `Welcome, ${role || 'technician'}` });
    } catch (err) {
      setToast({ type: 'error', message: err.response?.data?.detail || err.message });
    }
  };

  /**
   * Handle logout - clears auth state and localStorage
   */
  const handleLogout = () => {
    setAuth({ loggedIn: false, role: null, username: '' });
    localStorage.removeItem('auth_token');
    localStorage.removeItem('auth_role');
    localStorage.removeItem('auth_username');
    setSelectedProduct(null);
    setResult(null);
  };

  // ---------------------------------------------------------------------------
  // DIAGNOSIS FUNCTIONS
  // ---------------------------------------------------------------------------

  /**
   * Send fault description to AI for repair suggestion
   * Uses RAG to retrieve relevant context from documents
   */
  const handleDiagnose = async (isRetry = false, excludedSources = []) => {
    if (!selectedProduct || !faultDescription) {
      alert('Please select a product and describe the fault');
      return;
    }

    if (isRetry) {
      setRetryLoading(true);
    } else {
      setLoading(true);
    }
    
    // Reset feedback state for new diagnosis
    setFeedbackSubmitted(false);
    setShowFeedbackModal(false);

    try {
      const requestData = {
        part_number: selectedProduct.part_number,
        fault_description: faultDescription,
        language: language
      };
      
      // Add retry parameters if this is a retry
      if (isRetry && result?.diagnosis_id) {
        requestData.is_retry = true;
        requestData.retry_of = result.diagnosis_id;
        requestData.excluded_sources = excludedSources.length > 0 
          ? excludedSources 
          : result.sources?.map(s => s.source) || [];
      }

      const response = await api.post('/diagnose', requestData);

      setResult(response.data);
    } catch (error) {
      console.error('Error getting diagnosis:', error);
      setToast({ type: 'error', message: error.response?.data?.detail || error.message });
    } finally {
      setLoading(false);
      setRetryLoading(false);
    }
  };

  /**
   * Submit feedback for a diagnosis
   */
  const handleFeedback = async (feedbackType) => {
    if (!result?.diagnosis_id) {
      console.error('No diagnosis ID available');
      return;
    }
    
    if (feedbackType === 'negative') {
      setShowFeedbackModal(true);
      return;
    }
    
    // Positive feedback - submit with source relevance if provided
    try {
      const feedbackData = {
        diagnosis_id: result.diagnosis_id,
        feedback_type: 'positive'
      };
      
      // Include source relevance feedback if user rated any sources
      if (Object.keys(sourceRelevance).length > 0) {
        feedbackData.source_relevance = Object.entries(sourceRelevance).map(([source, relevant]) => ({
          source,
          relevant
        }));
      }
      
      await api.post('/diagnose/feedback', feedbackData);
      
      setFeedbackSubmitted(true);
      setToast({ type: 'info', message: language === 'tr' ? 'Geri bildiriminiz için teşekkürler! 👍' : 'Thanks for your feedback! 👍' });
    } catch (error) {
      console.error('Error submitting feedback:', error);
      setToast({ type: 'error', message: 'Failed to submit feedback' });
    }
  };

  /**
   * Submit negative feedback with reason
   */
  const submitNegativeFeedback = async (requestRetry = false) => {
    if (!result?.diagnosis_id) return;
    
    try {
      const feedbackData = {
        diagnosis_id: result.diagnosis_id,
        feedback_type: 'negative',
        negative_reason: feedbackReason || 'other',
        user_comment: feedbackComment
      };
      
      // Include source relevance feedback if user rated any sources
      if (Object.keys(sourceRelevance).length > 0) {
        feedbackData.source_relevance = Object.entries(sourceRelevance).map(([source, relevant]) => ({
          source,
          relevant
        }));
      }
      
      await api.post('/diagnose/feedback', feedbackData);
      
      setFeedbackSubmitted(true);
      setShowFeedbackModal(false);
      setFeedbackReason('');
      setFeedbackComment('');
      setSourceRelevance({});  // Reset source relevance
      
      if (requestRetry) {
        // Get alternative suggestion
        setToast({ type: 'info', message: language === 'tr' ? 'Yeni öneri alınıyor...' : 'Getting new suggestion...' });
        handleDiagnose(true, result.sources?.map(s => s.source) || []);
      } else {
        setToast({ type: 'info', message: language === 'tr' ? 'Geri bildiriminiz kaydedildi' : 'Feedback recorded' });
      }
    } catch (error) {
      console.error('Error submitting negative feedback:', error);
      setToast({ type: 'error', message: 'Failed to submit feedback' });
    }
  };

  // ---------------------------------------------------------------------------
  // UTILITY FUNCTIONS
  // ---------------------------------------------------------------------------

  /**
   * Handle product selection from list or dropdown
   * Supports both event objects and direct product objects
   */
  const handleProductSelect = (e) => {
    if (!e) return;
    
    // Direct product object (from grid click)
    if (typeof e === 'object' && e.part_number) {
      setSelectedProduct(e);
      return;
    }

    // Select dropdown change event
    const value = e.target ? e.target.value : e;
    const product = products.find(p => p.part_number === value);
    setSelectedProduct(product);
  };

  /**
   * Extract image URLs from product data
   * Handles various image field formats from API
   */
  const getImages = (p) => {
    if (!p) return [];
    
    // Filter out placeholder images
    const isValidImage = (url) => {
      if (!url || typeof url !== 'string') return false;
      const lower = url.toLowerCase();
      // Skip placeholder, default, and missing images
      if (lower.includes('placeholder') || lower.includes('default') || lower === '-') return false;
      return true;
    };
    
    // Handle array fields
    if (Array.isArray(p.image_url)) return p.image_url.filter(isValidImage);
    if (Array.isArray(p.images)) return p.images.filter(isValidImage);
    
    // Handle image_urls field (comma-separated string or array)
    if (p.image_urls) {
      if (Array.isArray(p.image_urls)) return p.image_urls.filter(isValidImage);
      if (typeof p.image_urls === 'string') {
        return p.image_urls.split(',').map(s => s.trim()).filter(isValidImage);
      }
    }
    
    // Handle single image_url string
    if (p.image_url && typeof p.image_url === 'string') {
      if (p.image_url.includes(',')) {
        return p.image_url.split(',').map(s => s.trim()).filter(isValidImage);
      }
      if (isValidImage(p.image_url)) {
        return [p.image_url];
      }
    }
    
    return [];
  };

  // ===========================================================================
  // RENDER FUNCTIONS - UI Components
  // ===========================================================================

  /**
   * Render Dashboard Tab Content
   */
  const renderDashboardTab = () => (
    <div className="dashboard-content">
      {dashboardLoading ? (
        <div className="dashboard-loading">
          <div className="spinner"></div>
          <p>Loading dashboard data...</p>
        </div>
      ) : !dashboardData ? (
        <div className="dashboard-empty">
          <p>Dashboard data not available yet. Make some diagnoses first!</p>
          <button className="btn-primary" onClick={loadDashboard}>🔄 Refresh</button>
        </div>
      ) : (
        <>
          {/* Overview Cards */}
          <div className="dashboard-overview">
            <div className="overview-card">
              <div className="overview-icon">🔍</div>
              <div className="overview-info">
                <span className="overview-value">{dashboardData.overview?.total_diagnoses || 0}</span>
                <span className="overview-label">Total Diagnoses</span>
              </div>
            </div>
            <div className="overview-card highlight">
              <div className="overview-icon">📅</div>
              <div className="overview-info">
                <span className="overview-value">{dashboardData.overview?.today_diagnoses || 0}</span>
                <span className="overview-label">Today</span>
              </div>
            </div>
            <div className="overview-card">
              <div className="overview-icon">📊</div>
              <div className="overview-info">
                <span className="overview-value">{dashboardData.overview?.week_diagnoses || 0}</span>
                <span className="overview-label">This Week</span>
              </div>
            </div>
            <div className="overview-card">
              <div className="overview-icon">👥</div>
              <div className="overview-info">
                <span className="overview-value">{dashboardData.overview?.active_users_week || 0}</span>
                <span className="overview-label">Active Users</span>
              </div>
            </div>
            <div className="overview-card">
              <div className="overview-icon">⚡</div>
              <div className="overview-info">
                <span className="overview-value">{((dashboardData.overview?.avg_response_time_ms || 0) / 1000).toFixed(1)}s</span>
                <span className="overview-label">Avg Response</span>
              </div>
            </div>
            <div className="overview-card">
              <div className="overview-icon">😊</div>
              <div className="overview-info">
                <span className="overview-value">{dashboardData.feedback?.satisfaction_rate || 0}%</span>
                <span className="overview-label">Satisfaction</span>
              </div>
            </div>
          </div>

          {/* Charts Row */}
          <div className="dashboard-charts">
            {/* Daily Trend */}
            <div className="chart-card">
              <h3>📈 Daily Trend (Last 7 Days)</h3>
              <div className="chart-bar-container">
                {dashboardData.daily_trend?.map((day, i) => (
                  <div key={i} className="chart-bar-item">
                    <div 
                      className="chart-bar" 
                      style={{ 
                        height: `${Math.max(day.count * 20, 10)}px`,
                        backgroundColor: day.count > 0 ? 'var(--primary)' : '#e0e0e0'
                      }}
                    >
                      <span className="bar-value">{day.count}</span>
                    </div>
                    <span className="bar-label">{day.day}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Confidence Breakdown */}
            <div className="chart-card">
              <h3>🎯 Confidence Breakdown</h3>
              <div className="confidence-chart">
                <div className="confidence-item">
                  <div className="confidence-bar-wrapper">
                    <div 
                      className="confidence-bar high" 
                      style={{ 
                        width: `${dashboardData.overview?.total_diagnoses > 0 
                          ? (dashboardData.confidence_breakdown?.high / dashboardData.overview.total_diagnoses * 100) 
                          : 0}%` 
                      }}
                    ></div>
                  </div>
                  <span className="confidence-label">High ({dashboardData.confidence_breakdown?.high || 0})</span>
                </div>
                <div className="confidence-item">
                  <div className="confidence-bar-wrapper">
                    <div 
                      className="confidence-bar medium" 
                      style={{ 
                        width: `${dashboardData.overview?.total_diagnoses > 0 
                          ? (dashboardData.confidence_breakdown?.medium / dashboardData.overview.total_diagnoses * 100) 
                          : 0}%` 
                      }}
                    ></div>
                  </div>
                  <span className="confidence-label">Medium ({dashboardData.confidence_breakdown?.medium || 0})</span>
                </div>
                <div className="confidence-item">
                  <div className="confidence-bar-wrapper">
                    <div 
                      className="confidence-bar low" 
                      style={{ 
                        width: `${dashboardData.overview?.total_diagnoses > 0 
                          ? (dashboardData.confidence_breakdown?.low / dashboardData.overview.total_diagnoses * 100) 
                          : 0}%` 
                      }}
                    ></div>
                  </div>
                  <span className="confidence-label">Low ({dashboardData.confidence_breakdown?.low || 0})</span>
                </div>
              </div>
            </div>

            {/* Feedback Stats */}
            <div className="chart-card">
              <h3>📝 Feedback Statistics</h3>
              <div className="feedback-stats-grid">
                <div className="feedback-stat positive">
                  <span className="feedback-icon">👍</span>
                  <span className="feedback-value">{dashboardData.feedback?.positive_feedback || 0}</span>
                  <span className="feedback-label">Positive</span>
                </div>
                <div className="feedback-stat negative">
                  <span className="feedback-icon">👎</span>
                  <span className="feedback-value">{dashboardData.feedback?.negative_feedback || 0}</span>
                  <span className="feedback-label">Negative</span>
                </div>
                <div className="feedback-stat learned">
                  <span className="feedback-icon">🧠</span>
                  <span className="feedback-value">{dashboardData.feedback?.learned_mappings || 0}</span>
                  <span className="feedback-label">Learned</span>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom Row - Lists */}
          <div className="dashboard-lists">
            {/* Top Products */}
            <div className="list-card">
              <h3>🏆 Top Diagnosed Products</h3>
              {dashboardData.top_products?.length > 0 ? (
                <div className="top-list">
                  {dashboardData.top_products.slice(0, 5).map((p, i) => (
                    <div key={i} className="top-item">
                      <span className="top-rank">#{i + 1}</span>
                      <div className="top-info">
                        <span className="top-name">{p.model || p.part_number}</span>
                        <span className="top-part">{p.part_number}</span>
                      </div>
                      <span className="top-count">{p.count}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="empty-list">No data yet</p>
              )}
            </div>

            {/* Top Faults */}
            <div className="list-card">
              <h3>🔧 Common Fault Keywords</h3>
              {dashboardData.top_faults?.length > 0 ? (
                <div className="fault-tags">
                  {dashboardData.top_faults.slice(0, 10).map((f, i) => (
                    <span key={i} className="fault-tag">
                      {f.keyword} <small>({f.count})</small>
                    </span>
                  ))}
                </div>
              ) : (
                <p className="empty-list">No data yet</p>
              )}
            </div>

            {/* System Info */}
            <div className="list-card">
              <h3>⚙️ System Status</h3>
              <div className="system-stats">
                <div className="system-stat">
                  <span className="system-label">Products in DB</span>
                  <span className="system-value">{dashboardData.system?.products_in_db || 0}</span>
                </div>
                <div className="system-stat">
                  <span className="system-label">Documents in RAG</span>
                  <span className="system-value">{dashboardData.system?.documents_in_vectordb || 0}</span>
                </div>
                <div className="system-stat">
                  <span className="system-label">RAG Engine</span>
                  <span className="system-value status-online">Online</span>
                </div>
                <div className="system-stat">
                  <span className="system-label">Retry Rate</span>
                  <span className="system-value">{dashboardData.overview?.retry_rate || 0}%</span>
                </div>
              </div>
            </div>
          </div>

          {/* Refresh Button */}
          <div className="dashboard-actions">
            <button className="btn-primary" onClick={loadDashboard}>
              🔄 Refresh Dashboard
            </button>
          </div>
        </>
      )}
    </div>
  );

  /**
   * Render Admin Panel
   * Includes: Dashboard, user management, document management tabs
   */
  const renderAdminPanel = () => (
    <div className="admin-layout">
      {/* Admin Tabs */}
      <div className="admin-tabs">
        <button 
          className={`admin-tab ${adminTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => setAdminTab('dashboard')}
        >
          📊 Dashboard
        </button>
        <button 
          className={`admin-tab ${adminTab === 'users' ? 'active' : ''}`}
          onClick={() => setAdminTab('users')}
        >
          👥 Users
        </button>
        <button 
          className={`admin-tab ${adminTab === 'documents' ? 'active' : ''}`}
          onClick={() => setAdminTab('documents')}
        >
          📚 Documents
        </button>
        <button 
          className={`admin-tab ${adminTab === 'maintenance' ? 'active' : ''}`}
          onClick={() => setAdminTab('maintenance')}
        >
          🛠️ Maintenance
        </button>
      </div>

      {/* Tab Content */}
      {adminTab === 'dashboard' && renderDashboardTab()}

      {adminTab === 'users' && (
        <div className="admin-tab-content">
          {/* Stats Overview */}
          <div className="card stats-card">
            <h2>📊 System Overview</h2>
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-icon">📦</span>
                <div className="stat-info">
                  <span className="stat-value">{stats?.products_in_db || 0}</span>
                  <span className="stat-label">Products</span>
                </div>
              </div>
              <div className="stat-item">
                <span className="stat-icon">📄</span>
                <div className="stat-info">
                  <span className="stat-value">{stats?.documents_in_vectordb || 0}</span>
                  <span className="stat-label">Documents</span>
                </div>
              </div>
              <div className="stat-item">
                <span className="stat-icon">👥</span>
                <div className="stat-info">
                  <span className="stat-value">{adminUsers.length}</span>
                  <span className="stat-label">Users</span>
                </div>
              </div>
              <div className="stat-item">
                <span className="stat-icon">🤖</span>
                <div className="stat-info">
                  <span className="stat-value status-online">Online</span>
                  <span className="stat-label">RAG Engine</span>
                </div>
              </div>
            </div>
          </div>

          {/* User Management */}
          <div className="card">
            <h2>👥 User Management</h2>
            
            <form onSubmit={handleAdminAddUser} className="add-user-form">
              <h3>Add New User</h3>
              <div className="form-row">
                <input
                  type="text"
                  placeholder="Username"
                  value={adminNewUser.username}
                  onChange={(e) => setAdminNewUser({ ...adminNewUser, username: e.target.value })}
                />
                <input
                  type="password"
                  placeholder="Password"
                  value={adminNewUser.password}
                  onChange={(e) => setAdminNewUser({ ...adminNewUser, password: e.target.value })}
                />
                <select
                  value={adminNewUser.role}
                  onChange={(e) => setAdminNewUser({ ...adminNewUser, role: e.target.value })}
                >
                  <option value="technician">Technician</option>
                  <option value="admin">Admin</option>
                </select>
                <button type="submit" className="btn-primary btn-sm">+ Add</button>
              </div>
            </form>

            <div className="users-list">
              <h3>Current Users</h3>
              {adminLoadingUsers ? (
                <p className="loading">Loading users...</p>
              ) : adminUsers.length === 0 ? (
                <p className="empty">No users found (endpoint may not be available)</p>
              ) : (
                <table className="users-table">
                  <thead>
                    <tr>
                      <th>Username</th>
                      <th>Role</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {adminUsers.map((user) => (
                      <tr key={user.username}>
                        <td>{user.username}</td>
                        <td>
                          <span className={`role-badge role-${user.role}`}>
                            {user.role}
                          </span>
                        </td>
                        <td>
                          {user.username !== auth.username && (
                            <button
                              className="btn-danger btn-sm"
                              onClick={() => handleAdminDeleteUser(user.username)}
                            >
                              Delete
                            </button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </div>
      )}

      {adminTab === 'documents' && (
        <div className="admin-tab-content">
          {/* Document Management for RAG */}
          <div className="card document-management">
            <h2>📚 RAG Document Management</h2>
            <p className="section-desc">Upload documents (PDF, Word, PowerPoint) to enhance the AI repair assistant knowledge base.</p>
            
            {/* Upload Section */}
            <div className="upload-section">
              <h3>Upload New Document</h3>
              <div className="upload-form">
                <select 
                  value={adminUploadType} 
                  onChange={(e) => setAdminUploadType(e.target.value)}
                  className="doc-type-select"
                >
                  <option value="manual">📖 Manual</option>
                  <option value="bulletin">📋 Service Bulletin</option>
                </select>
                <label className="file-upload-btn">
                  <input 
                    type="file" 
                    accept=".pdf,.docx,.doc,.pptx,.ppt" 
                    onChange={handleAdminUploadDocument}
                    disabled={adminUploading}
                  />
                  {adminUploading ? '⏳ Uploading...' : '📤 Select Document'}
                </label>
              </div>
            </div>

            {/* Documents List */}
            <div className="documents-list">
              <div className="docs-header">
                <h3>Uploaded Documents ({adminDocuments.length})</h3>
                <button 
                  className="btn-action btn-ingest"
                  onClick={handleAdminIngestDocuments}
                  disabled={adminIngesting || adminDocuments.length === 0}
                >
                  {adminIngesting ? '⏳ Processing...' : '🔄 Re-index All Documents'}
                </button>
              </div>
              
              {adminLoadingDocs ? (
                <p className="loading">Loading documents...</p>
              ) : adminDocuments.length === 0 ? (
                <div className="empty-docs">
                  <span>📂</span>
                  <p>No documents uploaded yet. Upload PDF, Word, or PowerPoint files to improve RAG responses.</p>
                </div>
              ) : (
                <table className="docs-table">
                  <thead>
                    <tr>
                      <th>Filename</th>
                      <th>Type</th>
                      <th>Format</th>
                      <th>Size</th>
                      <th>Uploaded</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {adminDocuments.map((doc) => {
                      const formatIcons = { pdf: '📕', docx: '📘', doc: '📘', pptx: '📙', ppt: '📙' };
                      const formatIcon = formatIcons[doc.format] || '📄';
                      return (
                        <tr key={`${doc.type}-${doc.filename}`}>
                          <td className="doc-name">
                            <span className="doc-icon">{formatIcon}</span>
                            {doc.filename}
                          </td>
                          <td>
                            <span className={`doc-type-badge ${doc.type}`}>{doc.type}</span>
                          </td>
                          <td className="doc-format">{(doc.format || 'pdf').toUpperCase()}</td>
                          <td>{(doc.size / 1024).toFixed(1)} KB</td>
                          <td>{new Date(doc.uploaded).toLocaleDateString()}</td>
                          <td>
                            <button
                              className="btn-danger btn-sm"
                              onClick={() => handleAdminDeleteDocument(doc)}
                            >
                              🗑️ Delete
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </div>
      )}

      {adminTab === 'maintenance' && (
        <div className="admin-tab-content">
          <div className="card">
            <h2>🛠️ Maintenance</h2>
            
            <div className="action-buttons">
              <button 
                className="btn-action"
                onClick={handleAdminScrape}
                disabled={adminScraping}
              >
                {adminScraping ? '⏳ Scraping...' : '🔄 Run Scraper'}
              </button>
              
              <button 
                className="btn-action"
                onClick={() => { loadProducts(); loadStats(); loadDashboard(); setToast({ type: 'info', message: 'Data refreshed' }); }}
              >
                🔄 Refresh All Data
              </button>
              
              <button 
                className="btn-action"
                onClick={() => window.open(`${API_BASE}/docs`, '_blank')}
              >
                📚 API Docs
              </button>
              
              <button 
                className="btn-action"
                onClick={() => window.open(`${API_BASE}/ui`, '_blank')}
              >
                🔧 Simple UI
              </button>
            </div>

            <div className="quick-stats">
              <h3>Quick Info</h3>
              <ul>
                <li><strong>API:</strong> {API_BASE}</li>
                <li><strong>User:</strong> {auth.username}</li>
                <li><strong>Role:</strong> {auth.role}</li>
              </ul>
            </div>
          </div>

          {/* Products Preview */}
          <div className="card">
            <h2>📦 Recent Products</h2>
            <div className="products-preview">
              {products.slice(0, 6).map((p) => (
                <div key={p.part_number} className="product-preview-card">
                  <div className="preview-img">
                    {getImages(p).length > 0 ? (
                      <img src={getImages(p)[0]} alt={p.model_name} />
                    ) : (
                      <span className="no-img">📷</span>
                    )}
                  </div>
                  <div className="preview-info">
                    <strong>{p.model_name}</strong>
                    <span>{p.part_number}</span>
                  </div>
                </div>
              ))}
            </div>
            <button 
              className="btn-secondary" 
              style={{ marginTop: '16px' }}
              onClick={() => setAuth({ ...auth, role: 'technician' })}
            >
              View All Products (Switch to Technician View)
            </button>
          </div>
        </div>
      )}
    </div>
  );

  /**
   * Render Technician Panel
   * Uses new TechWizard component for step-by-step diagnosis
   */
  const renderTechnicianPanel = () => (
    <TechWizard token={localStorage.getItem('auth_token')} />
  );

  /**
   * OLD Technician Panel Code (Archived)
   * Keeping for reference - replaced with TechWizard component
   */
  const renderTechnicianPanelOld = () => (
      <div className="technician-layout">
        {/* Left: Product Selection */}
        <section className="tech-left">
          <div className="card">
            <div className="card-header">
              <h2>📦 Products</h2>
              <div className="view-toggle">
                <button 
                  className={viewMode === 'grid' ? 'active' : ''} 
                  onClick={() => setViewMode('grid')}
                  title="Grid View"
                >
                  ⊞
                </button>
                <button 
                  className={viewMode === 'list' ? 'active' : ''} 
                  onClick={() => setViewMode('list')}
                  title="List View"
                >
                  ☰
                </button>
              </div>
            </div>

            {/* Search & Filters */}
            <div className="search-filters">
              <div className="search-box">
                <span className="search-icon">🔍</span>
                <input
                  type="search"
                  placeholder="Search by model or part number..."
                  value={search}
                  onChange={(e) => { setSearch(e.target.value); setPage(1); }}
                />
              </div>
              
              <div className="filter-row">
                <select 
                  value={filterSeries} 
                  onChange={(e) => { setFilterSeries(e.target.value); setPage(1); }}
                >
                  <option value="">All Series</option>
                  {facets.series.map(s => <option key={s} value={s}>{s}</option>)}
                </select>

                <label className="checkbox-filter">
                  <input
                    type="checkbox"
                    checked={filterWirelessOnly}
                    onChange={(e) => { setFilterWirelessOnly(e.target.checked); setPage(1); }}
                  />
                  Wireless Only
                </label>
              </div>
            </div>

            {/* Product Results Info */}
            <div className="results-info">
              <span>{filteredProducts.length} products found</span>
              {(search || filterSeries || filterWirelessOnly) && (
                <button 
                  className="clear-filters"
                  onClick={() => {
                    setSearch('');
                    setFilterSeries('');
                    setFilterWirelessOnly(false);
                    setPage(1);
                  }}
                >
                  Clear Filters
                </button>
              )}
            </div>

            {/* Product Grid/List */}
            <div className={`products-container ${viewMode}`}>
              {pagedProducts.map(product => (
                <div
                  key={product.part_number}
                  className={`product-card ${selectedProduct?.part_number === product.part_number ? 'selected' : ''}`}
                  onClick={() => handleProductSelect(product)}
                >
                  <div className="pc-img">
                    {getImages(product).length > 0 ? (
                      <img src={getImages(product)[0]} alt={product.model_name} />
                    ) : (
                      <span className="no-img">📷</span>
                    )}
                  </div>
                  <div className="pc-body">
                    <div className="pc-title">{product.model_name}</div>
                    <div className="pc-sub">{product.part_number}</div>
                    <div className="pc-tags">
                      {product.series_name && <span className="tag">{product.series_name}</span>}
                      {product.wireless_communication?.toLowerCase().includes('yes') && (
                        <span className="tag tag-wireless">📶 Wireless</span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="pagination">
                <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page <= 1}>
                  ← Prev
                </button>
                <span className="page-info">Page {page} of {totalPages}</span>
                <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page >= totalPages}>
                  Next →
                </button>
              </div>
            )}
          </div>
        </section>

        {/* Right: Diagnosis */}
        <section className="tech-right">
          {/* Selected Product Details */}
          {selectedProduct && (
            <div className="card selected-product-card">
              <h2>🔧 Selected Product</h2>
              <div className="selected-product-details">
                <div className="sp-image">
                  {getImages(selectedProduct).length > 0 ? (
                    <img src={getImages(selectedProduct)[0]} alt={selectedProduct.model_name} />
                  ) : (
                    <span className="no-img-lg">📷</span>
                  )}
                </div>
                <div className="sp-info">
                  <h3>{selectedProduct.model_name}</h3>
                  <div className="sp-details">
                    <div><strong>Part Number:</strong> {selectedProduct.part_number}</div>
                    <div><strong>Series:</strong> {selectedProduct.series_name || 'N/A'}</div>
                    <div><strong>Torque:</strong> {selectedProduct.min_torque || '?'} - {selectedProduct.max_torque || '?'}</div>
                    <div><strong>Output:</strong> {selectedProduct.output_drive || 'N/A'}</div>
                    <div><strong>Wireless:</strong> {selectedProduct.wireless_communication || 'N/A'}</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Diagnosis Form */}
          <div className="card diagnosis-card">
            <h2>🩺 Diagnose Problem</h2>
            
            {!selectedProduct && (
              <div className="select-product-hint">
                <span>👈</span>
                <p>Select a product from the list to start diagnosis</p>
              </div>
            )}

            <div className="form-group">
              <label>Fault Description *</label>
              <textarea
                value={faultDescription}
                onChange={(e) => setFaultDescription(e.target.value)}
                placeholder="Describe the problem in detail... (e.g., motor makes grinding noise, tool doesn't start, battery drains quickly)"
                rows="4"
                disabled={loading || !selectedProduct}
              />
            </div>

            <div className="form-group">
              <label>Response Language</label>
              <div className="language-selector">
                <button 
                  className={language === 'en' ? 'active' : ''}
                  onClick={() => setLanguage('en')}
                  disabled={loading}
                >
                  🇬🇧 English
                </button>
                <button 
                  className={language === 'tr' ? 'active' : ''}
                  onClick={() => setLanguage('tr')}
                  disabled={loading}
                >
                  🇹🇷 Türkçe
                </button>
              </div>
            </div>

            <button 
              className="btn-primary btn-diagnose"
              onClick={handleDiagnose}
              disabled={loading || !selectedProduct || !faultDescription}
            >
              {loading ? (
                <span className="spinner">⏳ Analyzing...</span>
              ) : (
                <>🔍 Get Repair Suggestion</>
              )}
            </button>
          </div>

          {/* Result */}
          {result && (
            <div className="card result-card">
              <div className="result-header">
                <h2>✅ Repair Suggestion</h2>
                <span className={`confidence confidence-${result.confidence}`}>
                  {result.confidence === 'high' && '🟢'}
                  {result.confidence === 'medium' && '🟡'}
                  {result.confidence === 'low' && '🔴'}
                  {result.confidence}
                </span>
              </div>

              <div className="result-content">
                <div className="suggestion-text">
                  {result.suggestion}
                </div>

                {result.sources && result.sources.length > 0 && (
                  <div className="sources-section">
                    <h4>📚 {result.language === 'tr' ? 'İlgili Dokümanlar' : 'Related Documents'} ({result.sources.length})</h4>
                    <p className="sources-hint">
                      {result.language === 'tr' 
                        ? 'Dokümanları değerlendirin ve faydalı olanları işaretleyin:' 
                        : 'Rate documents and mark helpful ones:'}
                    </p>
                    <div className="sources-cards">
                      {result.sources.slice(0, 5).map((source, idx) => (
                        <div key={idx} className={`source-card ${sourceRelevance[source.source] === true ? 'relevant' : sourceRelevance[source.source] === false ? 'irrelevant' : ''}`}>
                          <div className="source-info">
                            <span className="source-name">{source.source}</span>
                            <span className="source-similarity">{result.language === 'tr' ? 'Benzerlik' : 'Similarity'}: {source.similarity}</span>
                          </div>
                          <div className="source-actions">
                            {/* Relevance feedback buttons */}
                            {!feedbackSubmitted && (
                              <div className="relevance-buttons">
                                <button 
                                  className={`relevance-btn relevant ${sourceRelevance[source.source] === true ? 'active' : ''}`}
                                  onClick={() => setSourceRelevance(prev => ({...prev, [source.source]: true}))}
                                  title={result.language === 'tr' ? 'Alakalı' : 'Relevant'}
                                >
                                  ✓
                                </button>
                                <button 
                                  className={`relevance-btn irrelevant ${sourceRelevance[source.source] === false ? 'active' : ''}`}
                                  onClick={() => setSourceRelevance(prev => ({...prev, [source.source]: false}))}
                                  title={result.language === 'tr' ? 'Alakasız' : 'Not Relevant'}
                                >
                                  ✗
                                </button>
                              </div>
                            )}
                            <button 
                              className="open-doc-btn"
                              onClick={() => window.open(`${API_BASE}/documents/download/${encodeURIComponent(source.source)}`, '_blank')}
                            >
                              📄 {result.language === 'tr' ? 'Aç' : 'Open'}
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                    {result.sources.length > 5 && (
                      <details className="more-sources">
                        <summary>+{result.sources.length - 5} {result.language === 'tr' ? 'daha fazla kaynak' : 'more sources'}</summary>
                        <ul className="sources-list">
                          {result.sources.slice(5).map((source, idx) => (
                            <li key={idx} className={sourceRelevance[source.source] === true ? 'relevant' : sourceRelevance[source.source] === false ? 'irrelevant' : ''}>
                              <span>{source.source}</span>
                              <div className="source-actions-small">
                                {!feedbackSubmitted && (
                                  <div className="relevance-buttons-small">
                                    <button 
                                      className={`relevance-btn-small relevant ${sourceRelevance[source.source] === true ? 'active' : ''}`}
                                      onClick={() => setSourceRelevance(prev => ({...prev, [source.source]: true}))}
                                    >
                                      ✓
                                    </button>
                                    <button 
                                      className={`relevance-btn-small irrelevant ${sourceRelevance[source.source] === false ? 'active' : ''}`}
                                      onClick={() => setSourceRelevance(prev => ({...prev, [source.source]: false}))}
                                    >
                                      ✗
                                    </button>
                                  </div>
                                )}
                                <button 
                                  className="open-doc-btn-small"
                                  onClick={() => window.open(`${API_BASE}/documents/download/${encodeURIComponent(source.source)}`, '_blank')}
                                >
                                  {result.language === 'tr' ? 'Aç' : 'Open'}
                                </button>
                              </div>
                            </li>
                          ))}
                        </ul>
                      </details>
                    )}
                  </div>
                )}

                {/* Feedback Section */}
                <div className="feedback-section">
                  <h4>
                    {result.language === 'tr' ? 'Bu öneri işinize yaradı mı?' : 'Was this suggestion helpful?'}
                  </h4>
                  
                  {/* Source relevance summary */}
                  {!feedbackSubmitted && Object.keys(sourceRelevance).length > 0 && (
                    <div className="source-relevance-summary">
                      <span className="relevance-icon">📊</span>
                      {result.language === 'tr' 
                        ? `${Object.values(sourceRelevance).filter(v => v === true).length} alakalı, ${Object.values(sourceRelevance).filter(v => v === false).length} alakasız kaynak işaretlendi`
                        : `${Object.values(sourceRelevance).filter(v => v === true).length} relevant, ${Object.values(sourceRelevance).filter(v => v === false).length} irrelevant sources marked`}
                    </div>
                  )}
                  
                  {retryLoading ? (
                    <div className="retry-loading">
                      <div className="spinner"></div>
                      <p>{result.language === 'tr' ? 'Yeni öneri hazırlanıyor...' : 'Preparing new suggestion...'}</p>
                    </div>
                  ) : !feedbackSubmitted ? (
                    <div className="feedback-buttons">
                      <button 
                        className="feedback-btn positive"
                        onClick={() => handleFeedback('positive')}
                      >
                        <span>👍</span> {result.language === 'tr' ? 'Evet, Faydalı' : 'Yes, Helpful'}
                      </button>
                      <button 
                        className="feedback-btn negative"
                        onClick={() => handleFeedback('negative')}
                      >
                        <span>👎</span> {result.language === 'tr' ? 'Hayır, Farklı Öneri' : 'No, Try Different'}
                      </button>
                    </div>
                  ) : (
                    <div className="feedback-success">
                      <span>✅</span> {result.language === 'tr' ? 'Geri bildiriminiz kaydedildi!' : 'Feedback recorded!'}
                    </div>
                  )}
                </div>

                {/* Response time */}
                {result.response_time_ms && (
                  <div className="response-time">
                    ⚡ {result.language === 'tr' ? 'Yanıt süresi' : 'Response time'}: {(result.response_time_ms / 1000).toFixed(1)}s
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Feedback Modal */}
          {showFeedbackModal && (
            <div className="feedback-modal">
              <div className="modal-overlay" onClick={() => setShowFeedbackModal(false)}></div>
              <div className="modal-content" onClick={e => e.stopPropagation()}>
                <h4>
                  {result?.language === 'tr' ? 'Neden faydalı olmadı?' : 'Why wasn\'t it helpful?'}
                </h4>
                
                <div className="feedback-reasons">
                  <label className={`reason-option ${feedbackReason === 'wrong_product' ? 'selected' : ''}`}>
                    <input 
                      type="radio" 
                      name="reason" 
                      value="wrong_product"
                      checked={feedbackReason === 'wrong_product'}
                      onChange={(e) => setFeedbackReason(e.target.value)}
                    />
                    <span>{result?.language === 'tr' ? 'Yanlış ürün/parça önerildi' : 'Wrong product/part suggested'}</span>
                  </label>
                  <label className={`reason-option ${feedbackReason === 'wrong_fault_type' ? 'selected' : ''}`}>
                    <input 
                      type="radio" 
                      name="reason" 
                      value="wrong_fault_type"
                      checked={feedbackReason === 'wrong_fault_type'}
                      onChange={(e) => setFeedbackReason(e.target.value)}
                    />
                    <span>{result?.language === 'tr' ? 'Arıza tipi farklı' : 'Different fault type'}</span>
                  </label>
                  <label className={`reason-option ${feedbackReason === 'incomplete_info' ? 'selected' : ''}`}>
                    <input 
                      type="radio" 
                      name="reason" 
                      value="incomplete_info"
                      checked={feedbackReason === 'incomplete_info'}
                      onChange={(e) => setFeedbackReason(e.target.value)}
                    />
                    <span>{result?.language === 'tr' ? 'Eksik bilgi var' : 'Missing information'}</span>
                  </label>
                  <label className={`reason-option ${feedbackReason === 'incorrect_steps' ? 'selected' : ''}`}>
                    <input 
                      type="radio" 
                      name="reason" 
                      value="incorrect_steps"
                      checked={feedbackReason === 'incorrect_steps'}
                      onChange={(e) => setFeedbackReason(e.target.value)}
                    />
                    <span>{result?.language === 'tr' ? 'Adımlar yanlış' : 'Steps are incorrect'}</span>
                  </label>
                  <label className={`reason-option ${feedbackReason === 'other' ? 'selected' : ''}`}>
                    <input 
                      type="radio" 
                      name="reason" 
                      value="other"
                      checked={feedbackReason === 'other'}
                      onChange={(e) => setFeedbackReason(e.target.value)}
                    />
                    <span>{result?.language === 'tr' ? 'Diğer' : 'Other'}</span>
                  </label>
                </div>

                {feedbackReason === 'other' && (
                  <div className="custom-feedback-input">
                    <textarea
                      placeholder={result?.language === 'tr' ? 'Lütfen açıklayınız...' : 'Please explain...'}
                      value={feedbackComment}
                      onChange={(e) => setFeedbackComment(e.target.value)}
                      rows={3}
                    />
                  </div>
                )}

                <div className="modal-buttons">
                  <button 
                    className="modal-btn cancel"
                    onClick={() => setShowFeedbackModal(false)}
                  >
                    {result?.language === 'tr' ? 'İptal' : 'Cancel'}
                  </button>
                  <button 
                    className="modal-btn submit"
                    onClick={() => submitNegativeFeedback(true)}
                    disabled={!feedbackReason || retryLoading}
                  >
                    {retryLoading ? '...' : (result?.language === 'tr' ? '🔄 Yeni Öneri Al' : '🔄 Get New Suggestion')}
                  </button>
                </div>
              </div>
            </div>
          )}
        </section>
      </div>
    );
  

  // ===========================================================================
  // MAIN RENDER - Application Root Component
  // ===========================================================================
  
  // Show loading screen while checking authentication
  if (initializing) {
    return (
      <div className="app">
        <div className="init-loader">
          <div className="init-spinner"></div>
          <p>Loading Desoutter Repair Assistant...</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="app">
      <nav className="navbar">
        <div className="container navbar-inner">
          <div className="brand">
            <span className="logo">🔧</span>
            <span>Desoutter Repair Assistant</span>
          </div>
          <div className="nav-actions">
            {stats && (
              <span className="badge badge-soft">{stats.products_in_db} Products</span>
            )}
            {auth.role === 'admin' && (
              <a className="link" href={`${API_BASE}/docs`} target="_blank" rel="noreferrer">API Docs</a>
            )}
          </div>
        </div>
      </nav>

      <header className="header">
        <div className="container header-content">
          <div className="header-left">
            <div className="header-badge">
              <span className="badge-icon">🤖</span>
              <span className="badge-text">AI Powered</span>
            </div>
            <h1 className="header-title">
              Repair <span className="highlight">Assistant</span>
            </h1>
            <p className="header-subtitle">
              Smart diagnostics for Desoutter industrial tools
            </p>
            <div className="header-features">
              <span className="feature"><span className="feature-icon">⚡</span> Fast Analysis</span>
              <span className="feature"><span className="feature-icon">📚</span> RAG Technology</span>
              <span className="feature"><span className="feature-icon">🎯</span> Accurate Results</span>
            </div>
          </div>
          <div className="header-right">
            {stats && (
              <div className="stats-grid">
                <div className="stat-box">
                  <div className="stat-icon">📦</div>
                  <div className="stat-info">
                    <span className="stat-value">{stats.products_in_db}</span>
                    <span className="stat-title">Products</span>
                  </div>
                </div>
                <div className="stat-box">
                  <div className="stat-icon">📄</div>
                  <div className="stat-info">
                    <span className="stat-value">{stats.documents_in_vectordb}</span>
                    <span className="stat-title">Documents</span>
                  </div>
                </div>
                <div className="stat-box status-box">
                  <div className="status-indicator">
                    <span className="status-pulse"></span>
                  </div>
                  <div className="stat-info">
                    <span className="stat-value status-text">Online</span>
                    <span className="stat-title">System Status</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="container">
        {toast && (
          <div className={`toast toast-${toast.type}`} onAnimationEnd={() => setToast(null)}>
            {toast.message}
          </div>
        )}
        {!auth.loggedIn ? (
          <div className="login-wrap">
            <div className="card login-card">
              <h2>Sign In</h2>
              <form onSubmit={handleLogin} className="login-form">
                <div className="form-group">
                  <label>Username</label>
                  <input
                    type="text"
                    value={loginForm.username}
                    onChange={(e) => setLoginForm({ ...loginForm, username: e.target.value })}
                    placeholder="e.g. admin or technician"
                  />
                </div>
                <div className="form-group">
                  <label>Password</label>
                  <input
                    type="password"
                    value={loginForm.password}
                    onChange={(e) => setLoginForm({ ...loginForm, password: e.target.value })}
                    placeholder="••••••••"
                  />
                </div>
                <button type="submit" className="btn-primary">Sign In</button>
              </form>
              <p className="login-hint">Hint: usernames containing "admin" get admin role; others are technician (temporary)</p>
            </div>
          </div>
        ) : (
          <>
            <div className="role-bar">
              <span>Signed in as <strong>{auth.username}</strong> ({auth.role})</span>
              <button className="btn-secondary" onClick={handleLogout}>Sign Out</button>
            </div>
            {auth.role === 'admin' ? renderAdminPanel() : renderTechnicianPanel()}
          </>
        )}

      </main>

      <footer className="footer">
        <div className="container footer-content">
          <div className="footer-main">
            <p className="footer-brand">Powered by Ollama + ChromaDB + FastAPI</p>
            <p className="footer-infra">🏗️ Running on Proxmox AI Infrastructure</p>
          </div>
          <div className="footer-credits">
            <span>Developed by</span>
            <a href="https://github.com/fatihhbayramm" target="_blank" rel="noreferrer" className="footer-link">
              <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
            </a>
            <a href="https://www.linkedin.com/in/fatihhbayramm/" target="_blank" rel="noreferrer" className="footer-link">
              <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>
            </a>
            <span className="footer-author">adentechio</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
