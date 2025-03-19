package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/jwtauth/v5"
	"github.com/google/uuid"
	_ "github.com/lib/pq"
	"golang.org/x/crypto/bcrypt"
)

type User struct {
	ID           uuid.UUID `json:"id"`
	Email        string    `json:"email"`
	PasswordHash string    `json:"-"`
	Name         string    `json:"name"`
	Institution  string    `json:"institution,omitempty"`
	OrcidID      string    `json:"orcid_id,omitempty"`
	GoogleID     string    `json:"google_id,omitempty"`
	LinkedInID   string    `json:"linkedin_id,omitempty"`
	CreatedAt    time.Time `json:"created_at"`
	UpdatedAt    time.Time `json:"updated_at"`
}

type LoginRequest struct {
	Email    string `json:"email"`
	Password string `json:"password"`
}

type RegisterRequest struct {
	Email       string `json:"email"`
	Password    string `json:"password"`
	Name        string `json:"name"`
	Institution string `json:"institution,omitempty"`
}

type AuthResponse struct {
	Token string `json:"token"`
	User  User   `json:"user"`
}

var tokenAuth *jwtauth.JWTAuth
var db *sql.DB

func init() {
	// Inicjalizacja JWT
	tokenAuth = jwtauth.New("HS256", []byte(os.Getenv("JWT_SECRET_KEY")), nil)

	// Połączenie z bazą danych
	connStr := fmt.Sprintf("postgres://%s:%s@%s/%s?sslmode=disable",
		os.Getenv("DB_USER"),
		os.Getenv("DB_PASSWORD"),
		os.Getenv("DB_HOST"),
		os.Getenv("DB_NAME"))

	var err error
	db, err = sql.Open("postgres", connStr)
	if err != nil {
		log.Fatal(err)
	}

	if err = db.Ping(); err != nil {
		log.Fatal(err)
	}
}

func main() {
	r := chi.NewRouter()

	// Middleware
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)

	// Publiczne endpointy
	r.Group(func(r chi.Router) {
		r.Get("/health", func(w http.ResponseWriter, r *http.Request) {
			w.Write([]byte("Auth Service is healthy"))
		})

		r.Post("/login", handleLogin)
		r.Post("/register", handleRegister)
		r.Post("/oauth/google", handleGoogleOAuth)
		r.Post("/oauth/orcid", handleOrcidOAuth)
		r.Post("/oauth/linkedin", handleLinkedInOAuth)
	})

	// Prywatne endpointy (wymagają autoryzacji)
	r.Group(func(r chi.Router) {
		r.Use(jwtauth.Verifier(tokenAuth))
		r.Use(jwtauth.Authenticator)

		r.Get("/user", handleGetUser)
		r.Put("/user", handleUpdateUser)
	})

	port := "8081"
	fmt.Printf("Auth Service listening on port %s\n", port)
	log.Fatal(http.ListenAndServe(":"+port, r))
}

func handleLogin(w http.ResponseWriter, r *http.Request) {
	var req LoginRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Sprawdzanie poświadczeń użytkownika
	var user User
	var passwordHash string
	err := db.QueryRow("SELECT id, email, password_hash, name, institution, created_at, updated_at FROM users WHERE email = $1",
		req.Email).Scan(&user.ID, &user.Email, &passwordHash, &user.Name, &user.Institution, &user.CreatedAt, &user.UpdatedAt)
	if err != nil {
		http.Error(w, "Invalid credentials", http.StatusUnauthorized)
		return
	}

	// Weryfikacja hasła
	if err := bcrypt.CompareHashAndPassword([]byte(passwordHash), []byte(req.Password)); err != nil {
		http.Error(w, "Invalid credentials", http.StatusUnauthorized)
		return
	}

	// Generowanie tokenu JWT
	_, tokenString, _ := tokenAuth.Encode(map[string]interface{}{
		"user_id": user.ID.String(),
		"exp":     time.Now().Add(24 * time.Hour).Unix(),
	})

	response := AuthResponse{
		Token: tokenString,
		User:  user,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func handleRegister(w http.ResponseWriter, r *http.Request) {
	var req RegisterRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Sprawdzanie czy użytkownik już istnieje
	var exists bool
	err := db.QueryRow("SELECT EXISTS(SELECT 1 FROM users WHERE email = $1)", req.Email).Scan(&exists)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if exists {
		http.Error(w, "User already exists", http.StatusConflict)
		return
	}

	// Hashowanie hasła
	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(req.Password), bcrypt.DefaultCost)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Tworzenie użytkownika
	var user User
	user.ID = uuid.New()
	user.Email = req.Email
	user.Name = req.Name
	user.Institution = req.Institution
	user.CreatedAt = time.Now()
	user.UpdatedAt = time.Now()

	_, err = db.Exec(`
		INSERT INTO users (id, email, password_hash, name, institution, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
	`, user.ID, user.Email, string(hashedPassword), user.Name, user.Institution, user.CreatedAt, user.UpdatedAt)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Generowanie tokenu JWT
	_, tokenString, _ := tokenAuth.Encode(map[string]interface{}{
		"user_id": user.ID.String(),
		"exp":     time.Now().Add(24 * time.Hour).Unix(),
	})

	response := AuthResponse{
		Token: tokenString,
		User:  user,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

func handleGoogleOAuth(w http.ResponseWriter, r *http.Request) {
	// Implementacja OAuth z Google Scholar
	w.Write([]byte("Google OAuth will be implemented here"))
}

func handleOrcidOAuth(w http.ResponseWriter, r *http.Request) {
	// Implementacja OAuth z ORCID
	w.Write([]byte("ORCID OAuth will be implemented here"))
}

func handleLinkedInOAuth(w http.ResponseWriter, r *http.Request) {
	// Implementacja OAuth z LinkedIn
	w.Write([]byte("LinkedIn OAuth will be implemented here"))
}

func handleGetUser(w http.ResponseWriter, r *http.Request) {
	// Pobranie ID użytkownika z tokenu JWT
	_, claims, _ := jwtauth.FromContext(r.Context())
	userID := claims["user_id"].(string)

	// Pobranie danych użytkownika
	var user User
	err := db.QueryRow(`
		SELECT id, email, name, institution, orcid_id, google_id, linkedin_id, created_at, updated_at
		FROM users WHERE id = $1
	`, userID).Scan(&user.ID, &user.Email, &user.Name, &user.Institution, &user.OrcidID, &user.GoogleID, &user.LinkedInID, &user.CreatedAt, &user.UpdatedAt)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(user)
}

func handleUpdateUser(w http.ResponseWriter, r *http.Request) {
	// Pobranie ID użytkownika z tokenu JWT
	_, claims, _ := jwtauth.FromContext(r.Context())
	userID := claims["user_id"].(string)

	var user User
	if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Aktualizacja danych użytkownika
	_, err := db.Exec(`
		UPDATE users SET name = $1, institution = $2, updated_at = $3 WHERE id = $4
	`, user.Name, user.Institution, time.Now(), userID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
}