#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <number>\n", argv[0]);
        return 1;
    }

    int x = atoi(argv[1]);
    printf("Input value: %d\n", x);

    // First level: Check if negative
    int is_negative = (x < 0);
    printf("Is negative: %d\n", is_negative);

    if (is_negative) {
        // Second level: Check if very negative
        int is_very_negative = (x < -10);
        printf("Is very negative: %d\n", is_very_negative);

        if (is_very_negative) {
            printf("Very negative branch\n");
        } else {
            printf("Slightly negative branch\n");
        }
    } else {
        // Second level: Check if zero
        int is_zero = (x == 0);
        printf("Is zero: %d\n", is_zero);

        if (is_zero) {
            printf("Zero branch\n");
        } else {
            // Third level: Check if very positive
            int is_very_positive = (x > 10);
            printf("Is very positive: %d\n", is_very_positive);

            if (is_very_positive) {
                printf("Very positive branch\n");
            } else {
                printf("Slightly positive branch\n");
            }
        }
    }

    return 0;
}