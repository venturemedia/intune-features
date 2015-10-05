//  Copyright © 2015 Venture Media. All rights reserved.

import Foundation
import Surge

struct Feature {
    let peakCount: Int
    let noteCount: Int
    
    var label: Int = -1
    var peaks: [Point] {
        willSet {
            precondition(newValue.count == peakCount)
        }
    }
    var bands: [Double] {
        willSet {
            precondition(newValue.count == noteCount)
        }
    }
    var RMS: Double
    
    init(peakCount: Int, noteCount: Int) {
        self.peakCount = peakCount
        self.noteCount = noteCount
        
        peaks = [Point](count: peakCount, repeatedValue: Point())
        bands = [Double](count: noteCount, repeatedValue: 0.0)
        RMS = 0.0
    }

    static func dataSize(peakCount: Int, noteCount: Int) -> Int {
        let peakSize = 2 * peakCount
        let bandsSize = noteCount
        let RMSSize = 1
        return peakSize + bandsSize + RMSSize
    }

    func data() -> [Double] {
        var data = [Double]()
        for point in peaks {
            data.appendContentsOf([point.x / 10000.0, point.y])
        }
        data.appendContentsOf(bands)
        data.append(RMS)
        return data
    }
}
