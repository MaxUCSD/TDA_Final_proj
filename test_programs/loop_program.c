#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <number>\n", argv[0]);
        return 1;
    }

    int x = atoi(argv[1]);
    printf("Input value: %d\n", x);

    // Handle negative numbers differently
    if (x < 0) {
        printf("Negative input, taking absolute value\n");
        x = -x;
    }

    // Cap the loop to prevent excessive execution
    if (x > 10) {
        printf("Large input, capping at 10\n");
        x = 10;
    }

    printf("Processing %d iterations\n", x);
    
    int sum = 0;
    for (int i = 0; i < x; i++) {
        printf("Iteration %d: ", i);
        
        // Nested branching within the loop
        if (i % 2 == 0) {
            printf("even processing\n");
            sum += i * 2;
        } else {
            printf("odd processing\n");
            sum += i * 3;
        }
        
        // Additional branching based on accumulated sum
        if (sum > 20) {
            printf("  High sum reached: %d\n", sum);
        }
    }
    
    printf("Final sum: %d\n", sum);
    
    // Final output classification
    if (sum == 0) {
        printf("Zero result\n");
    } else if (sum < 10) {
        printf("Low result\n");
    } else if (sum < 50) {
        printf("Medium result\n");
    } else {
        printf("High result\n");
    }
    
    return 0;
}