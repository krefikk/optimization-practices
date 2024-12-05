#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 28
#define V (N * N + 1)
#define EPOCHS 1000
#define LR 0.001

float tanh_activation(float z) {
    return tanh(z);
}

float tanh_derivative(float z) {
    float t = tanh(z);
    return 1 - t * t;
}

float model_output(float w[V], float x[V]) {
    float z = 0.0;
    for (int i = 0; i < V; i++) {
        z += w[i] * x[i];
    }
    return tanh_activation(z);
}

float compute_loss(float w[V], float X[][V], float y[], int size) {
    float loss = 0.0;
    for (int i = 0; i < size; i++) {
        float pred = model_output(w, X[i]);
        loss += pow(pred - y[i], 2);
    }
    return loss / size;
}

// Gradient Descent
void gradient_descent(float w[V], float X[][V], float y[], int size) {
    float gradients[V] = { 0.0 };
    for (int i = 0; i < size; i++) {
        float pred = model_output(w, X[i]);
        float error = pred - y[i];
        for (int j = 0; j < V; j++) {
            gradients[j] += error * tanh_derivative(pred) * X[i][j];
        }
    }
    for (int j = 0; j < V; j++) {
        w[j] -= LR * gradients[j] / size;
    }
}

// Stochastic Gradient Descent
void stochastic_gradient_descent(float w[V], float X[][V], float y[], int size) {
    for (int i = 0; i < size; i++) {
        float pred = model_output(w, X[i]);
        float error = pred - y[i];
        for (int j = 0; j < V; j++) {
            w[j] -= LR * error * tanh_derivative(pred) * X[i][j];
        }
    }
}

// Adam
void adam_optimizer(float w[V], float X[][V], float y[], int size, int epoch) {
    float m[V] = { 0.0 }, v[V] = { 0.0 };
    float beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
    for (int i = 0; i < size; i++) {
        float pred = model_output(w, X[i]);
        float error = pred - y[i];
        for (int j = 0; j < V; j++) {
            float grad = error * tanh_derivative(pred) * X[i][j];
            m[j] = beta1 * m[j] + (1 - beta1) * grad;
            v[j] = beta2 * v[j] + (1 - beta2) * grad * grad;

            // Bias correction
            float m_hat = m[j] / (1 - pow(beta1, epoch + 1));
            float v_hat = v[j] / (1 - pow(beta2, epoch + 1));

            // Weight fix
            w[j] -= LR * m_hat / (sqrt(v_hat) + epsilon);
        }
    }
}

float compute_accuracy(float w[V], float X[][V], float y[], int size) {
    int correct_predictions = 0;
    for (int i = 0; i < size; i++) {
        float pred = model_output(w, X[i]);
        // Compute accuracy with correct / total
        int predicted_class = (pred >= 0.0) ? 1 : -1;
        if (predicted_class == y[i]) {
            correct_predictions++;
        }
    }
    return (float)correct_predictions / size;
}

void train_with_logging(float w[V], float X_train[][V], float y_train[], int train_size,
    float X_test[][V], float y_test[], int test_size,
    int algorithm, FILE* log_file) {
    clock_t epoch_start, epoch_end;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        epoch_start = clock();

        if (algorithm == 0) gradient_descent(w, X_train, y_train, train_size);
        if (algorithm == 1) stochastic_gradient_descent(w, X_train, y_train, train_size);
        if (algorithm == 2) adam_optimizer(w, X_train, y_train, train_size, epoch);

        epoch_end = clock();

        // Eðitim kaybý
        float train_loss = compute_loss(w, X_train, y_train, train_size);
        // Eðitim doðruluðunu hesapla
        float train_accuracy = compute_accuracy(w, X_train, y_train, train_size);
        // Test doðruluðu
        float test_accuracy = compute_accuracy(w, X_test, y_test, test_size);
        double time_taken = (double)(epoch_end - epoch_start) / CLOCKS_PER_SEC;

        // Log dosyasýna aðýrlýklarý ve test baþarýmýný ekleyin
        fprintf(log_file, "%d,%f,%f,%f,%f", epoch, train_loss, test_accuracy, train_accuracy, time_taken);
        for (int i = 0; i < V; i++) {
            fprintf(log_file, ",%f", w[i]);
        }
        fprintf(log_file, "\n");
    }
}

int main() {
    srand(time(NULL));
    int train_size = 200, test_size = 50;
    float X_train[200][V], y_train[200];
    float X_test[50][V], y_test[50];
    float w[V], initial_weights[5][V];

    /*
    // Rastgele baþlangýç aðýrlýklarý üret
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < V; j++) {
            initial_weights[i][j] = ((float)rand() / RAND_MAX) * 0.01;
        }
    }
    */
    // Rastgele baþlangýç aðýrlýklarý yerine sabit deðerler
    float values[5] = { 0.1, 0.01, 0.001, 0.0001, 0.00001 };
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < V; j++) {
            initial_weights[i][j] = values[i];
        }
    }

    for (int i = 0; i < train_size; i++) {
        for (int j = 0; j < V - 1; j++) {
            X_train[i][j] = ((float)rand() / RAND_MAX);
        }
        X_train[i][V - 1] = 1.0;
        y_train[i] = (i < train_size / 2) ? 1.0 : -1.0;
    }
    for (int i = 0; i < test_size; i++) {
        for (int j = 0; j < V - 1; j++) {
            X_test[i][j] = ((float)rand() / RAND_MAX);
        }
        X_test[i][V - 1] = 1.0;
        y_test[i] = (i < test_size / 2) ? 1.0 : -1.0;
    }

    // 3 algoritma için 5 baþlangýç aðýrlýðý ile eðitim yap ve logla
    for (int alg = 0; alg < 3; alg++) {
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < V; j++) w[j] = initial_weights[i][j];
            char filename[50];
            sprintf(filename, "algorithm_%d_weights_w%.5f.csv", alg, initial_weights[i][0]);
            FILE* log_file = fopen(filename, "w");
            fprintf(log_file, "Epoch,TrainLoss,TestAccuracy,TrainAccuracy,Time");
            for (int i = 0; i < V; i++) {
                fprintf(log_file, ",w%d", i);
            }
            fprintf(log_file, "\n");
            train_with_logging(w, X_train, y_train, train_size, X_test, y_test, test_size, alg, log_file);
            fclose(log_file);
        }
    }
    return 0;
}