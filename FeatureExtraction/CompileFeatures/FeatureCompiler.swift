//  Copyright © 2015 Venture Media. All rights reserved.

import Foundation

import AudioKit
import FeatureExtraction
import HDF5
import Surge

typealias Point = Surge.Point<Double>

class FeatureCompiler {
    let fileListPath = "file.txt"
    let rootPath = ""
    let fileType = "caf"
    let ratio = 10

    let sampleRate = 44100
    let sampleCount = 32768

    let startNote = 24
    let peakCount = 10
    let notes = 24...96

    
    let fft: FFT
    let fb: Double
    var labelShape = [UInt64]()

    init(sampleCount: Int) {
        fft = FFT(inputLength: sampleCount)
        fb = Double(sampleRate) / Double(sampleCount)
    }
    
    func compileFeatures() {
        let fileCollector = FileCollector(noteRange: notes, fileType: fileType)
        let features = generateFeatures(fileCollector.buildExamples())
        splitAndWriteFeatures(features)
    }
    
    func generateFeatures(examples: [FileCollector.Example]) -> [FeatureSet] {
        var features = [FeatureSet]()
        var data = [Double](count: sampleCount, repeatedValue: 0.0)
        for example in examples {
            var featureSet = FeatureSet(peakCount: peakCount, noteCount: notes.count)
            featureSet.label = example.label
            
            let audioFile = AudioFile(filePath: example.filePath)!
            assert(audioFile.sampleRate == 44100)
            
            audioFile.readFrames(&data, count: sampleCount)
            
            let psd = sqrt(fft.forwardMags(data))
            
            let fftPoints = (0..<psd.count).map{ Point(x: fb * Double($0), y: psd[$0]) }
            let peaks = PeakExtractor.process(fftPoints).sort{ $0.y > $1.y }
            if peaks.count >= peakCount {
                featureSet.peaks = Array(peaks[0..<peakCount])
            }
            else {
                featureSet.peaks = [Point](count: peakCount, repeatedValue: Point())
                featureSet.peaks.replaceRange((0..<peaks.count), with: peaks)
            }
                
            let bands = BandExtractor.process(spectrumData: psd, startNote: startNote, numberOfNotes: notes.count, baseFrequency: fb)
            featureSet.bands = bands
            
            features.append(featureSet)
        }
        return features
    }

    func splitAndWriteFeatures(features: [FeatureSet]) {
        let split = features.count/ratio
        
        let testingFeatures = features[0..<split]
        writeFeatures(testingFeatures, fileName: "testing.h5")
        
        let trainingFeatures = features[split..<features.count]
        writeFeatures(trainingFeatures, fileName: "training.h5")
    }
    
    func writeFeatures(features: ArraySlice<FeatureSet>, fileName: String) {
        guard let hdf5File = HDF5.File.create(fileName, mode: File.CreateMode.Truncate) else {
            fatalError("Could not create HDF5 dataset.")
        }
        let dataType = HDF5.Datatype.copy(type: .Double)
        let labelType = HDF5.Datatype.copy(type: .Int)
        
        let hdf5DataDataspace = Dataspace(dims: [UInt64(features.count)])
        let hdf5Data = HDF5.Dataset.create(file: hdf5File, name: "data", datatype: dataType, dataspace: hdf5DataDataspace)
        
        let hdf5LabelDataspace = Dataspace(dims: labelShape)
        let hdf5Labels = HDF5.Dataset.create(file: hdf5File, name: "labels", datatype: labelType, dataspace: hdf5LabelDataspace)
        
        for featureSet in features {
            assert(hdf5Labels.writeInt([featureSet.label]))
            assert(hdf5Data.writeDouble(featureSet.data()))
        }
    }

}

