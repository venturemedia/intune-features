//  Copyright © 2015 Venture Media. All rights reserved.

import Foundation

extension String {
    mutating func appendPathComponent(component: String) {
        let strippedComponent: String
        if let first = component.characters.first where first == "/" {
            strippedComponent = component.substringFromIndex(component.startIndex.advancedBy(1))
        } else {
            strippedComponent = component
        }
        
        if let last = self.characters.last where last == "/" {
            self += strippedComponent
        } else {
            self += "/" + strippedComponent
        }
    }
}

func buildPathFromParts(parts: [String]) -> String {
    guard var path = parts.first else {
        return ""
    }
    
    for part in parts[1..<parts.count] {
        path.appendPathComponent(part)
    }
    return path
}
