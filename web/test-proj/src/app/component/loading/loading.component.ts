import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-loading',
  templateUrl: './loading.component.html',
  styleUrl: './loading.component.scss'
})
export class LoadingComponent {
  @Input() message: string = 'Detecting'; // 동적으로 설정되는 글자
  dots: string = ''; // 점점점의 상태
  private dotCount: number = 0;

  ngOnInit() {
    setInterval(() => {
      this.dotCount = (this.dotCount + 1) % 4; // 0, 1, 2, 3 반복
      this.dots = '.'.repeat(this.dotCount);  // 점을 dotCount만큼 생성
    }, 500); // 500ms마다 업데이트
  }
}
