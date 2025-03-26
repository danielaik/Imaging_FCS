package fiji.plugin.imaging_fcs.imfcs.utils;

import java.util.Iterator;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * Represents a range of integers with a specified step.
 * The range is defined by a start value, an end value, and a step.
 * Provides methods to check if a value is within the range and to get the length of the range.
 */
public class Range {
    private final int start;
    private final int end;
    private final int step;

    /**
     * Constructs a Range object with the specified start, end, and step values.
     *
     * @param start The starting value of the range.
     * @param end   The ending value of the range.
     * @param step  The step value between each element in the range.
     * @throws IllegalArgumentException if the range is invalid (i.e., step is non-positive for an ascending range or
     *                                  non-negative for a descending range).
     */
    public Range(int start, int end, int step) {
        this.start = start;
        this.end = end;
        this.step = step;

        checkValidRange();
    }

    /**
     * Validates the range to ensure it is logically consistent.
     * Throws an IllegalArgumentException if the range is invalid.
     */
    private void checkValidRange() {
        if (start > end && step >= 0 || start < end && step <= 0) {
            throw new IllegalArgumentException("Illegal range");
        }
    }

    /**
     * Checks if a given value is within the range.
     *
     * @param value The value to check.
     * @return True if the value is within the range, false otherwise.
     */
    public boolean contains(int value) {
        return start <= value && value <= end;
    }

    /**
     * Calculates the length of the range.
     *
     * @return The number of elements in the range.
     */
    public int length() {
        if ((step > 0 && start > end) || (step < 0 && start < end)) {
            return 0;
        }

        return Math.abs((end - start) / step) + 1;
    }

    /**
     * Generates a stream of integers from start to end with the specified step.
     *
     * @return A stream of integers representing the range.
     */
    public Stream<Integer> stream() {
        return StreamSupport.stream(Spliterators.spliteratorUnknownSize(new Iterator<Integer>() {
            private int current = start;

            @Override
            public boolean hasNext() {
                return step > 0 ? current < end : current > end;
            }

            @Override
            public Integer next() {
                int value = current;
                current += step;
                return value;
            }
        }, Spliterator.ORDERED), false);
    }

    public int getStart() {
        return start;
    }

    public int getEnd() {
        return end;
    }

    public int getStep() {
        return step;
    }
}
