import { Component, EventEmitter, Input, Output } from '@angular/core';

@Component({
  selector: 'app-chip',
  templateUrl: './chip.component.html',
  styleUrl: './chip.component.scss'
})
export class ChipComponent {
  @Input() items: string[] = []; // 목록을 입력받는 배열
  @Input() isSelectable: boolean = true; // 선택 가능 여부
  @Input() title: string = ''; // 제목
  @Output() selectionChange = new EventEmitter<string[]>(); // 선택된 목록 전달 이벤트

  selectedItems: Set<string> = new Set(); // 선택된 아이템 저장

  // Chip 클릭 이벤트: 선택/취소 상태 토글
  toggleSelection(item: string): void {
    if (!this.isSelectable) return; // 선택 불가능 시 무시
    if (this.selectedItems.has(item)) {
      this.selectedItems.delete(item);
    } else {
      this.selectedItems.add(item);
    }
    this.emitSelectionChange();
  }

  // 선택된 목록을 부모 컴포넌트로 전달
  private emitSelectionChange(): void {
    this.selectionChange.emit(Array.from(this.selectedItems));
  }

  // 선택 여부 확인
  isSelected(item: string): boolean {
    return this.selectedItems.has(item);
  }
}
