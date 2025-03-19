package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/cors"
	"github.com/go-chi/jwtauth/v5"
)

var tokenAuth *jwtauth.JWTAuth

func init() {
	// Inicjalizacja JWT - w produkcji klucz powinien być w zmiennych środowiskowych
	tokenAuth = jwtauth.New("HS256", []byte(os.Getenv("JWT_SECRET_KEY")), nil)
}

// ProxyMiddleware przekierowuje żądania do odpowiednich mikroserwisów
func ProxyMiddleware(serviceURL string) func(next http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Tutaj implementacja przekierowania żądania
			// W praktyce użylibyśmy biblioteki jak httputil.ReverseProxy
			
			// Przykładowa logika:
			fmt.Printf("Routing request to: %s\n", serviceURL)
			next.ServeHTTP(w, r)
		})
	}
}

func main() {
	r := chi.NewRouter()

	// Podstawowe middleware
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)
	r.Use(middleware.Timeout(60 * time.Second))

	// Konfiguracja CORS
	r.Use(cors.Handler(cors.Options{
		AllowedOrigins:   []string{"*"}, // W produkcji należy ograniczyć
		AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type"},
		ExposedHeaders:   []string{"Link"},
		AllowCredentials: true,
		MaxAge:           300,
	}))

	// Publiczne endpointy (bez autoryzacji)
	r.Group(func(r chi.Router) {
		r.Get("/health", func(w http.ResponseWriter, r *http.Request) {
			w.Write([]byte("API Gateway is healthy"))
		})

		// Routing do Auth Service dla endpointów logowania/rejestracji
		r.Route("/auth", func(r chi.Router) {
			r.Use(ProxyMiddleware("http://auth-service:8081"))
			r.Post("/login", func(w http.ResponseWriter, r *http.Request) {
				// Przekierowanie do Auth Service
			})
			r.Post("/register", func(w http.ResponseWriter, r *http.Request) {
				// Przekierowanie do Auth Service
			})
		})
	})

	// Chronione endpointy (z autoryzacją)
	r.Group(func(r chi.Router) {
		// Weryfikacja JWT
		r.Use(jwtauth.Verifier(tokenAuth))
		r.Use(jwtauth.Authenticator)

		// Routing do Research Service
		r.Route("/research", func(r chi.Router) {
			r.Use(ProxyMiddleware("http://research-service:8082"))
			r.Get("/", func(w http.ResponseWriter, r *http.Request) {
				// Przekierowanie do Research Service
			})
		})

		// Routing do Model Hosting Service
		r.Route("/models", func(r chi.Router) {
			r.Use(ProxyMiddleware("http://model-hosting-service:8083"))
			r.Get("/", func(w http.ResponseWriter, r *http.Request) {
				// Przekierowanie do Model Hosting Service
			})
		})

		// Routing do Forum/Chat Service
		r.Route("/forum", func(r chi.Router) {
			r.Use(ProxyMiddleware("http://forum-chat-service:8084"))
			r.Get("/", func(w http.ResponseWriter, r *http.Request) {
				// Przekierowanie do Forum/Chat Service
			})
		})

		// Routing do Recommendation Service
		r.Route("/recommendations", func(r chi.Router) {
			r.Use(ProxyMiddleware("http://recommendation-service:8085"))
			r.Get("/", func(w http.ResponseWriter, r *http.Request) {
				// Przekierowanie do Recommendation Service
			})
		})

		// Routing do Monetization Service
		r.Route("/monetization", func(r chi.Router) {
			r.Use(ProxyMiddleware("http://monetization-service:8086"))
			r.Get("/", func(w http.ResponseWriter, r *http.Request) {
				// Przekierowanie do Monetization Service
			})
		})
	})

	port := "8080"
	fmt.Printf("API Gateway listening on port %s\n", port)
	log.Fatal(http.ListenAndServe(":"+port, r))
}