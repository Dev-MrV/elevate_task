const API_BASE = 'http://127.0.0.1:8000/api';

const recommendForm = document.getElementById('recommend-form');
const loadingOverlay = document.getElementById('loading-overlay');
const statusBadge = document.getElementById('status-badge');
const resultsGrid = document.getElementById('results-grid');
const emptyState = document.getElementById('empty-state');
const movieList = document.getElementById('movie-list');
const alphaInput = document.getElementById('alpha');
const alphaVal = document.getElementById('alpha-val');

// Update alpha slider value display
alphaInput.addEventListener('input', (e) => {
    alphaVal.textContent = parseFloat(e.target.value).toFixed(1);
});

// Fetch movie titles for autocomplete on load
async function fetchMovies() {
    try {
        const res = await fetch(`${API_BASE}/movies`);
        if (!res.ok) throw new Error('Failed to fetch movies');
        const data = await res.json();
        
        data.movies.forEach(title => {
            const option = document.createElement('option');
            option.value = title;
            movieList.appendChild(option);
        });
        
    } catch (error) {
        console.error('Error fetching autocomplete data:', error);
    }
}

// Fetch random movies for initial screen
async function initRandomMovie() {
    loadingOverlay.classList.remove('hidden');
    loadingOverlay.classList.add('flex');
    statusBadge.textContent = "Loading Recommendations...";
    statusBadge.className = "px-3 py-1 bg-brand-purple/20 text-brand-purple border border-brand-purple/50 rounded-full text-xs font-semibold animate-pulse";
    
    try {
        const res = await fetch(`${API_BASE}/random`);
        if (res.ok) {
            const data = await res.json();
            renderRecommendations(data.recommendations);
            
            statusBadge.textContent = "Ready";
            statusBadge.className = "px-3 py-1 bg-green-500/20 text-green-400 border border-green-500/50 rounded-full text-xs font-semibold";
        }
    } catch (err) {
        console.error("Failed to fetch random movies", err);
    } finally {
        loadingOverlay.classList.add('hidden');
        loadingOverlay.classList.remove('flex');
    }
}

// Handle Form Submission
recommendForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const searchInput = document.getElementById('movie-search').value;
    if (!searchInput) {
        alert("Please enter a movie title.");
        return;
    }
    
    // Gather checked industries
    const industryCheckboxes = document.querySelectorAll('input[name="industry"]:checked');
    const industries = Array.from(industryCheckboxes).map(cb => cb.value);
    
    if (industries.length === 0) {
        alert("Please select at least one industry.");
        return;
    }
    
    const alpha = parseFloat(alphaInput.value);
    
    // UI Loading State
    loadingOverlay.classList.remove('hidden');
    loadingOverlay.classList.add('flex');
    statusBadge.textContent = "Computing Hybrid Matrix...";
    statusBadge.className = "px-3 py-1 bg-brand-purple/20 text-brand-purple border border-brand-purple/50 rounded-full text-xs font-semibold animate-pulse";
    
    try {
        const res = await fetch(`${API_BASE}/recommend`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                title: searchInput,
                industries: industries,
                alpha: alpha,
                top_n: 5
            })
        });
        
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Recommendation failed');
        }
        
        const data = await res.json();
        renderRecommendations(data.recommendations);
        
        statusBadge.textContent = "Match Complete";
        statusBadge.className = "px-3 py-1 bg-green-500/20 text-green-400 border border-green-500/50 rounded-full text-xs font-semibold";
        
    } catch (error) {
        console.error(error);
        alert(`Engine Error: ${error.message}`);
        statusBadge.textContent = "Error";
        statusBadge.className = "px-3 py-1 bg-red-500/20 text-red-400 border border-red-500/50 rounded-full text-xs font-semibold";
    } finally {
        loadingOverlay.classList.add('hidden');
        loadingOverlay.classList.remove('flex');
    }
});

function renderRecommendations(movies) {
    // Hide empty state
    emptyState.classList.add('hidden');
    
    // Clear grid
    resultsGrid.innerHTML = '';
    
    if (movies.length === 0) {
        resultsGrid.innerHTML = '<p class="text-gray-400 col-span-full text-center py-10">No matches found for these exact constraints.</p>';
        return;
    }
    
    const template = document.getElementById('card-template');
    
    movies.forEach(movie => {
        const clone = template.content.cloneNode(true);
        
        const cardTitle = clone.querySelector('.card-title');
        const cardPoster = clone.querySelector('.card-poster');
        const cardGenres = clone.querySelector('.card-genres');
        const langBadge = clone.querySelector('.lang-badge');
        const matchScore = clone.querySelector('.match-score');
        
        cardTitle.textContent = movie.title;
        cardTitle.title = movie.title;
        cardPoster.src = movie.poster_path;
        cardGenres.textContent = movie.genres;
        
        // Format Industry Badge
        const langMap = { 'ml': 'Malayalam', 'ta': 'Tamil', 'hi': 'Hindi', 'te': 'Telugu', 'en': 'Hollywood' };
        langBadge.textContent = langMap[movie.original_language] || movie.original_language.toUpperCase();
        
        // Color code match score
        const score = movie.match_score;
        matchScore.textContent = `${score}% Match`;
        
        if (score >= 80) {
            matchScore.classList.add('text-green-400', 'border-green-400/50');
        } else if (score >= 50) {
            matchScore.classList.add('text-yellow-400', 'border-yellow-400/50');
        } else {
            matchScore.classList.add('text-gray-300', 'border-gray-600');
        }
        
        resultsGrid.appendChild(clone);
    });
}

// Init
fetchMovies();
initRandomMovie();