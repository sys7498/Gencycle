import { Injectable, NgZone } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class SpeechRecognitionService {
  private recognition: SpeechRecognition;
  public isListening = false;

  constructor(private zone: NgZone) {
    const SpeechRecognition = window.SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      throw new Error('SpeechRecognition is not supported in this browser.');
    }

    this.recognition = new SpeechRecognition();
    this.recognition.lang = 'en-US';
    this.recognition.continuous = false; // Set to true if you want continuous recognition
    this.recognition.interimResults = false; // Set to true if you want intermediate results
  }

  startListening(): Promise<string> {
    return new Promise((resolve, reject) => {
      this.isListening = true;
      this.recognition.start();

      this.recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        this.zone.run(() => resolve(transcript));
      };

      this.recognition.onerror = (event) => {
        this.zone.run(() => reject(event.error));
      };

      this.recognition.onend = () => {
        this.isListening = false;
      };
    });
  }

  stopListening(): void {
    if (this.isListening) {
      this.recognition.stop();
    }
  }
}
