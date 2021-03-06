// Copyright © 2016 Venture Media Labs.
//
// This file is part of IntuneFeatures. The full IntuneFeatures copyright
// notice, including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Foundation
import Upsurge

public class PeakExtractor {
    let configuration: Configuration

    public init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func process(input: [Point], rms: Double) -> [Point] {
        let peaks = findPeaks(input)
        return filterPeaks(peaks, rms: rms)
    }

    func findPeaks(input: [Point]) -> [Point] {
        var peaks = [Point]()

        for i in 1...input.count-2 {
            let peak = input[i]
            if input[i-1].y <= peak.y && peak.y >= input[i+1].y {
                peaks.append(peak)
            }
        }

        return peaks
    }

    func filterPeaks(input: [Point], rms: Double) -> [Point] {
        let peaks = filterPeaksByHeight(input, rms: rms)
        return choosePeaks(peaks)
    }

    func filterPeaksByHeight(input: [Point], rms: Double) -> [Point] {
        return input.filter { (peak: Point) -> Bool in
            return peak.y > configuration.peakHeightCutoffMultiplier * rms
        }
    }

    func choosePeaks(input: [Point]) -> [Point] {
        var chosenPeaks = [Point]()

        var currentPeakRange = 0.0...0.0
        for peak in input {
            if currentPeakRange.contains(peak.x) {
                if let lastPeak = chosenPeaks.last where lastPeak.y < peak.y {
                    chosenPeaks.removeLast()
                    chosenPeaks.append(peak)
                    currentPeakRange = binCutoffRange(peak.x)
                }
            } else {
                chosenPeaks.append(peak)
                currentPeakRange = binCutoffRange(peak.x)
            }
        }

        return chosenPeaks
    }

    func binCutoffRange(freq: Double) -> ClosedInterval<Double> {
        let note = freqToNote(freq)

        let upperBound = noteToFreq(note + configuration.minimumPeakDistance)
        let lowerBound = noteToFreq(note - configuration.minimumPeakDistance)

        return lowerBound...upperBound
    }

}
